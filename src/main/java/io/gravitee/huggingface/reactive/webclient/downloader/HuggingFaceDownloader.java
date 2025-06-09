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
package io.gravitee.huggingface.reactive.webclient.downloader;

import io.gravitee.huggingface.reactive.webclient.client.VertxHuggingFaceClientRx;
import io.gravitee.huggingface.reactive.webclient.exception.ModelDownloadFailedException;
import io.gravitee.huggingface.reactive.webclient.exception.ModelFileNotFoundException;
import io.gravitee.resource.ai_model.api.ModelFetcher;
import io.gravitee.resource.ai_model.api.model.ModelFile;
import io.gravitee.resource.ai_model.api.model.ModelFileType;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.file.OpenOptions;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.file.AsyncFile;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HuggingFaceDownloader implements ModelFetcher {

    private static final Logger log = LoggerFactory.getLogger(HuggingFaceDownloader.class);
    private final Vertx vertx;

    private final VertxHuggingFaceClientRx modelDownloader;
    private final FetchModelConfig config;

    public HuggingFaceDownloader(Vertx vertx, FetchModelConfig config) {
        this(vertx, config, new VertxHuggingFaceClientRx(HuggingFaceWebClientFactory.createDefaultClient(vertx)));
    }

    public HuggingFaceDownloader(Vertx vertx, FetchModelConfig config, VertxHuggingFaceClientRx client) {
        this.vertx = vertx;
        this.config = config;
        this.modelDownloader = client;
    }

    @Override
    public Single<Map<ModelFileType, String>> fetchModel() {
        return modelDownloader
            .listModelFiles(config.modelName())
            .toList()
            .flatMapPublisher(availableFiles ->
                Flowable
                    .fromIterable(config.modelFiles())
                    .flatMapSingle(modelFile -> {
                        if (!availableFiles.contains(modelFile.name())) {
                            return Single.error(
                                new ModelFileNotFoundException(
                                    String.format(
                                        "Model file not found on HuggingFace: %s for model: %s",
                                        modelFile.name(),
                                        config.modelName()
                                    )
                                )
                            );
                        }

                        Path outputPath = config.modelDirectory().resolve(modelFile.name());

                        return vertx
                            .fileSystem()
                            .rxExists(outputPath.toString())
                            .flatMap(exists -> {
                                if (exists) {
                                    log.info("Skipping download; file already exists: {}", modelFile.name());
                                    return Single.just(Map.entry(modelFile.type(), outputPath.toString()));
                                }

                                boolean isFileInSubDirectory = outputPath.toString().contains("/");

                                return (
                                    isFileInSubDirectory ? buildSubDirectoryAndOpenFile(outputPath) : openFile(outputPath)
                                ).flatMap(file ->
                                        modelDownloader
                                            .downloadModelFile(config.modelName(), modelFile.name(), file)
                                            .doFinally(file::close)
                                            .andThen(Single.just(Map.entry(modelFile.type(), outputPath.toString())))
                                            .onErrorResumeNext(err ->
                                                Single.error(
                                                    new ModelDownloadFailedException(
                                                        String.format("Download failed for model: %s", modelFile.name()),
                                                        err
                                                    )
                                                )
                                            )
                                    )
                                    .doOnSuccess(resp -> log.info("Download completed successfully: {}", modelFile.name()))
                                    .doOnError(err -> log.error("Download failed", err));
                            });
                    })
            )
            .collect(() -> new EnumMap<>(ModelFileType.class), (map, entry) -> map.put(entry.getKey(), entry.getValue()));
    }

    private Single<AsyncFile> buildSubDirectoryAndOpenFile(Path outputPath) {
        return vertx.fileSystem().mkdirs(outputPath.getParent().toString()).andThen(openFile(outputPath));
    }

    private Single<AsyncFile> openFile(Path outputPath) {
        return vertx.fileSystem().rxOpen(outputPath.toString(), new OpenOptions().setCreate(true).setWrite(true).setTruncateExisting(true));
    }

    public record FetchModelConfig(String modelName, List<ModelFile> modelFiles, Path modelDirectory) {}
}
