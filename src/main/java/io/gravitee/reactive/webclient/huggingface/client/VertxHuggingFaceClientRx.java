/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.reactive.webclient.huggingface.client;

import io.gravitee.reactive.webclient.api.ModelFileInfo;
import io.gravitee.reactive.webclient.api.ModelInfo;
import io.gravitee.reactive.webclient.api.SafetensorsInfo;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.core.streams.WriteStream;
import io.vertx.rxjava3.ext.web.client.HttpRequest;
import io.vertx.rxjava3.ext.web.client.WebClient;
import io.vertx.rxjava3.ext.web.codec.BodyCodec;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VertxHuggingFaceClientRx implements HuggingFaceClientRx {

    private final Logger log = LoggerFactory.getLogger(VertxHuggingFaceClientRx.class);

    private static final String SIBLINGS_KEY = "siblings";
    private static final String RFILENAME_KEY = "rfilename";
    private static final String SIZE_KEY = "size";
    private static final String SAFETENSORS_KEY = "safetensors";
    private static final String PARAMETERS_KEY = "parameters";
    private static final String TOTAL_KEY = "total";

    private static final String HF_REPO_BASE_URL = "/%s/resolve/main/%s";

    private final WebClient client;
    private final String hfToken;

    public VertxHuggingFaceClientRx(WebClient client) {
        this(client, null);
    }

    public VertxHuggingFaceClientRx(WebClient client, String hfToken) {
        this.client = client;
        this.hfToken = (hfToken != null && !hfToken.isBlank()) ? hfToken : null;
    }

    @Override
    public Flowable<String> listModelFiles(String modelName) {
        return fetchModelInfo(modelName)
            .flattenAsFlowable(info -> info.files().stream().map(ModelFileInfo::name).toList())
            .doOnError(t ->
                log.error("An unexpected error has occurred while listing model repository [{}]: {}", modelName, t.getMessage(), t)
            )
            .onErrorComplete();
    }

    @Override
    public Single<ModelInfo> fetchModelInfo(String modelName) {
        var request = authorize(client.request(HttpMethod.GET, "/api/models/" + modelName));
        return request
            .putHeader("Accept", "application/json")
            .rxSend()
            .map(response -> {
                JsonObject body = response.body().toJsonObject();
                return parseModelInfo(modelName, body);
            })
            .doOnError(t -> log.error("Failed to fetch model info for [{}]: {}", modelName, t.getMessage(), t));
    }

    @Override
    public Single<JsonObject> fetchFileAsJson(String modelName, String filePath) {
        String url = String.format(HF_REPO_BASE_URL, modelName, filePath);
        var request = authorize(client.request(HttpMethod.GET, url));
        return request
            .putHeader("Accept", "application/json")
            .followRedirects(true)
            .rxSend()
            .map(response -> {
                if (response.statusCode() >= 400) {
                    throw new IllegalStateException(
                        String.format(
                            "Failed to fetch %s from [%s]: HTTP %d %s",
                            filePath,
                            modelName,
                            response.statusCode(),
                            response.statusMessage()
                        )
                    );
                }
                return response.body().toJsonObject();
            })
            .doOnError(t -> log.error("Failed to fetch {} for [{}]: {}", filePath, modelName, t.getMessage(), t));
    }

    @Override
    public Completable downloadModelFile(String modelName, String fileName, WriteStream<Buffer> file) {
        log.debug("Downloading file [{}] from model [{}]", fileName, modelName);
        var request = authorize(client.request(HttpMethod.GET, String.format(HF_REPO_BASE_URL, modelName, fileName)));
        return request
            .addQueryParam("download", "true")
            .followRedirects(true)
            .as(BodyCodec.pipe(file))
            .rxSend()
            .flatMapCompletable(response -> {
                if (response.statusCode() >= 400) {
                    return Completable.error(
                        new IllegalStateException(
                            String.format(
                                "Failed to download file: %s, status: %s, message: %s, body: %s",
                                fileName,
                                response.statusCode(),
                                response.statusMessage(),
                                response.bodyAsString()
                            )
                        )
                    );
                }
                return Completable.complete();
            })
            .doOnComplete(() -> log.info("Downloaded model file [{}] successfully", fileName))
            .doOnError(err -> log.error("Failed to download [{}]: {}", fileName, err.getMessage(), err));
    }

    // --- private helpers ---

    private <T> HttpRequest<T> authorize(HttpRequest<T> request) {
        if (hfToken != null) {
            request.putHeader("Authorization", "Bearer " + hfToken);
        }
        return request;
    }

    private static ModelInfo parseModelInfo(String modelName, JsonObject body) {
        boolean gated = body.getBoolean("gated", false);
        boolean isPrivate = body.getBoolean("private", false);

        JsonArray siblings = body.getJsonArray(SIBLINGS_KEY, new JsonArray());
        List<ModelFileInfo> files = siblings
            .stream()
            .map(obj -> {
                JsonObject o = (JsonObject) obj;
                String name = o.getString(RFILENAME_KEY, "");
                long size = o.getLong(SIZE_KEY, -1L);
                return new ModelFileInfo(name, size);
            })
            .toList();

        SafetensorsInfo safetensors = parseSafetensorsInfo(body.getJsonObject(SAFETENSORS_KEY));

        return new ModelInfo(modelName, gated, isPrivate, files, safetensors);
    }

    private static SafetensorsInfo parseSafetensorsInfo(JsonObject safetensorsJson) {
        if (safetensorsJson == null) {
            return null;
        }
        long total = safetensorsJson.getLong(TOTAL_KEY, 0L);
        JsonObject parametersJson = safetensorsJson.getJsonObject(PARAMETERS_KEY, new JsonObject());
        Map<String, Long> parameters = parametersJson
            .stream()
            .collect(Collectors.toMap(Map.Entry::getKey, e -> ((Number) e.getValue()).longValue()));
        return new SafetensorsInfo(parameters, total);
    }

    private static List<String> getFileNames(JsonArray siblings) {
        return siblings.stream().map(obj -> ((JsonObject) obj).getString(RFILENAME_KEY)).toList();
    }
}
