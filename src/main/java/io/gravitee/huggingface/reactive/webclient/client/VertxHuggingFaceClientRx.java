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
package io.gravitee.huggingface.reactive.webclient.client;

import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.core.streams.WriteStream;
import io.vertx.rxjava3.ext.web.client.WebClient;
import io.vertx.rxjava3.ext.web.codec.BodyCodec;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VertxHuggingFaceClientRx implements HuggingFaceClientRx {

    private final Logger log = LoggerFactory.getLogger(VertxHuggingFaceClientRx.class);
    private static final String SIBLINGS_KEY = "siblings";
    private static final String RFILENAME_KEY = "rfilename";

    private static final String HF_REPO_BASE_URL = "/%s/resolve/main/%s";

    private final WebClient client;

    public VertxHuggingFaceClientRx(WebClient client) {
        this.client = client;
    }

    @Override
    public Flowable<String> listModelFiles(String modelName) {
        var request = client.request(HttpMethod.GET, "/api/models/" + modelName);
        return request
            .putHeader("Accept", "application/json")
            .rxSend()
            .map(response -> response.body().toJsonObject().getJsonArray(SIBLINGS_KEY))
            .flattenAsFlowable(VertxHuggingFaceClientRx::getFileNames)
            .doOnError(t -> log.error("An unexpected error has occurred while list model repository [{}]: {}", modelName, t.getMessage(), t)
            )
            .onErrorComplete();
    }

    @Override
    public Completable downloadModelFile(String modelName, String fileName, WriteStream<Buffer> file) {
        log.debug("Downloading file [{}] from model [{}]", fileName, modelName);
        var request = client.request(HttpMethod.GET, String.format(HF_REPO_BASE_URL, modelName, fileName));
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

    private static List<String> getFileNames(JsonArray siblings) {
        return siblings.stream().map(obj -> ((JsonObject) obj).getString(RFILENAME_KEY)).toList();
    }
}
