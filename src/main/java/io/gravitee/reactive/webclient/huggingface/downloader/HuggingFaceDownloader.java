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
package io.gravitee.reactive.webclient.huggingface.downloader;

import io.gravitee.reactive.webclient.api.FetchModelConfig;
import io.gravitee.reactive.webclient.api.ModelFetcher;
import io.gravitee.reactive.webclient.api.ModelFile;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.gravitee.reactive.webclient.huggingface.client.VertxHuggingFaceClientRx;
import io.gravitee.reactive.webclient.huggingface.exception.ModelDownloadFailedException;
import io.gravitee.reactive.webclient.huggingface.exception.ModelFileNotFoundException;
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

    public HuggingFaceDownloader(Vertx vertx) {
        this(vertx, new VertxHuggingFaceClientRx(HuggingFaceWebClientFactory.createDefaultClient(vertx)));
    }

    public HuggingFaceDownloader(Vertx vertx, VertxHuggingFaceClientRx client) {
        this.vertx = vertx;
        this.modelDownloader = client;
    }

    @Override
    public Single<Map<ModelFileType, String>> fetchModel(FetchModelConfig config) {
        return Flowable.fromIterable(config.modelFiles())
            .flatMapSingle(modelFile -> {
                var outputPath = config.modelDirectory().resolve(modelFile.name());
                return vertx
                    .fileSystem()
                    .rxExists(outputPath.toString())
                    .map(exists -> new FileStatus(modelFile, outputPath, exists));
            })
            .toList()
            .flatMap(statuses -> {
                if (statuses.stream().allMatch(FileStatus::exists)) {
                    log.info("All model files already exist locally, skipping HuggingFace listing");
                    var result = new EnumMap<ModelFileType, String>(ModelFileType.class);
                    statuses.forEach(s -> result.put(s.modelFile().type(), s.outputPath().toString()));
                    return Single.just(result);
                }
                return downloadWithHuggingFaceVerification(config, statuses);
            });
    }

    private Single<Map<ModelFileType, String>> downloadWithHuggingFaceVerification(FetchModelConfig config, List<FileStatus> statuses) {
        return modelDownloader
            .listModelFiles(config.modelName())
            .toList()
            .flatMapPublisher(availableFiles ->
                Flowable.fromIterable(statuses).flatMapSingle(status -> {
                    if (status.exists()) {
                        log.info("Skipping download; file already exists: {}", status.modelFile().name());
                        return Single.just(Map.entry(status.modelFile().type(), status.outputPath().toString()));
                    }
                    if (!availableFiles.contains(status.modelFile().name())) {
                        return Single.error(
                            new ModelFileNotFoundException(
                                String.format(
                                    "Model file not found on HuggingFace: %s for model: %s",
                                    status.modelFile().name(),
                                    config.modelName()
                                )
                            )
                        );
                    }

                    boolean isFileInSubDirectory = status.outputPath().toString().contains("/");

                    return (
                        isFileInSubDirectory ? buildSubDirectoryAndOpenFile(status.outputPath()) : openFile(status.outputPath())
                    ).flatMap(file ->
                            modelDownloader
                                .downloadModelFile(config.modelName(), status.modelFile().name(), file)
                                .doFinally(file::close)
                                .andThen(Single.just(Map.entry(status.modelFile().type(), status.outputPath().toString())))
                                .onErrorResumeNext(err ->
                                    Single.error(
                                        new ModelDownloadFailedException(
                                            String.format("Download failed for model: %s", status.modelFile().name()),
                                            err
                                        )
                                    )
                                )
                        )
                        .doOnSuccess(resp -> log.info("Download completed successfully: {}", status.modelFile().name()))
                        .doOnError(err -> log.error("Download failed", err));
                })
            )
            .collect(() -> new EnumMap<>(ModelFileType.class), (map, entry) -> map.put(entry.getKey(), entry.getValue()));
    }

    private record FileStatus(ModelFile modelFile, Path outputPath, boolean exists) {}

    private Single<AsyncFile> buildSubDirectoryAndOpenFile(Path outputPath) {
        return vertx.fileSystem().mkdirs(outputPath.getParent().toString()).andThen(openFile(outputPath));
    }

    private Single<AsyncFile> openFile(Path outputPath) {
        return vertx.fileSystem().rxOpen(outputPath.toString(), new OpenOptions().setCreate(true).setWrite(true).setTruncateExisting(true));
    }
}
