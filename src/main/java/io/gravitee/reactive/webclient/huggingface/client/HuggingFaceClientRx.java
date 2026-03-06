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

import io.gravitee.reactive.webclient.api.ModelInfo;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.json.JsonObject;
import io.vertx.rxjava3.core.buffer.Buffer;
import io.vertx.rxjava3.core.streams.WriteStream;

public interface HuggingFaceClientRx {
    Flowable<String> listModelFiles(String modelName);

    /**
     * Fetches full model metadata from {@code GET /api/models/{modelName}},
     * including file list with sizes, gated flag, and safetensors parameter counts.
     *
     * <p>Returns a {@link Single} that emits {@link ModelInfo} on success.
     * On any HTTP or network error the Single terminates with an error signal —
     * callers should handle it (e.g. {@code .onErrorReturn(e -> null)}) if they
     * want a soft-fail behaviour.
     */
    Single<ModelInfo> fetchModelInfo(String modelName);

    /**
     * Fetches a single file from a model repository and returns its contents
     * as a parsed {@link JsonObject}.
     *
     * <p>Useful for retrieving {@code config.json}, {@code tokenizer_config.json},
     * etc. without downloading to disk.
     *
     * @param modelName  the model identifier, e.g. {@code "meta-llama/Llama-3-8B"}
     * @param filePath   relative path within the repo, e.g. {@code "config.json"}
     * @return a {@link Single} emitting the parsed JSON, or an error signal on failure
     */
    Single<JsonObject> fetchFileAsJson(String modelName, String filePath);

    Completable downloadModelFile(String modelName, String fileName, WriteStream<Buffer> file);
}
