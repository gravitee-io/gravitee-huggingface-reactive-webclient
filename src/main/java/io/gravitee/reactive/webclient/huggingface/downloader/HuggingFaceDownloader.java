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
import io.gravitee.reactive.webclient.api.ModelFileInfo;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.gravitee.reactive.webclient.api.ModelInfo;
import io.gravitee.reactive.webclient.huggingface.client.VertxHuggingFaceClientRx;
import io.gravitee.reactive.webclient.huggingface.exception.ModelDownloadFailedException;
import io.gravitee.reactive.webclient.huggingface.exception.ModelFileNotFoundException;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.file.OpenOptions;
import io.vertx.core.json.JsonObject;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.file.AsyncFile;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HuggingFaceDownloader implements ModelFetcher {

    private static final Logger log = LoggerFactory.getLogger(HuggingFaceDownloader.class);

    private final Vertx vertx;
    private final VertxHuggingFaceClientRx modelDownloader;

    /**
     * Creates a downloader with no HF token (public models only).
     */
    public HuggingFaceDownloader(Vertx vertx) {
        this(vertx, new VertxHuggingFaceClientRx(HuggingFaceWebClientFactory.createDefaultClient(vertx)));
    }

    /**
     * Creates a downloader with an optional HF token for gated/private models.
     *
     * @param hfToken Bearer token for HuggingFace Hub authentication, or {@code null}.
     */
    public static HuggingFaceDownloader withToken(Vertx vertx, String hfToken) {
        return new HuggingFaceDownloader(
            vertx,
            new VertxHuggingFaceClientRx(HuggingFaceWebClientFactory.createDefaultClient(vertx), hfToken)
        );
    }

    public HuggingFaceDownloader(Vertx vertx, VertxHuggingFaceClientRx client) {
        this.vertx = vertx;
        this.modelDownloader = client;
    }

    @Override
    public Single<Map<ModelFileType, String>> fetchModel(FetchModelConfig config) {
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

    /**
     * Fetches all model files whose names match the given predicate, downloading
     * each matching file into the target directory.
     *
     * <p>This is the auto-discovery path used by llama.cpp: the endpoint config
     * only supplies a model name (e.g. {@code "bartowski/Llama-3.1-8B-GGUF"})
     * and a filename glob/pattern; the downloader discovers which files to fetch
     * at runtime rather than requiring an explicit list.
     *
     * <p>Files that already exist locally are skipped (same behaviour as
     * {@link #fetchModel(FetchModelConfig)}).
     *
     * @param modelName   The HuggingFace model repository, e.g.
     *                    {@code "bartowski/Llama-3.1-8B-GGUF"}.
     * @param directory   Local directory to download into.
     * @param fileFilter  Predicate applied to each remote filename; only matching
     *                    files are downloaded.
     * @return A {@link Single} emitting the list of local paths that were fetched
     *         (already-existing files are included).
     */
    public Single<List<String>> fetchAllModelFiles(String modelName, Path directory, Predicate<String> fileFilter) {
        return modelDownloader
            .fetchModelInfo(modelName)
            .flatMapPublisher(info -> {
                List<String> matchingFiles = info.files().stream().map(ModelFileInfo::name).filter(fileFilter).toList();

                if (matchingFiles.isEmpty()) {
                    log.warn("No files matched the filter for model [{}]", modelName);
                }

                return Flowable.fromIterable(matchingFiles);
            })
            .flatMapSingle(fileName -> downloadFile(modelName, directory, fileName))
            .toList();
    }

    /**
     * Fetches full model metadata (file list, gated flag, safetensors info)
     * without downloading any weights. Useful for memory pre-flight checks.
     *
     * @param modelName The HuggingFace model repository identifier.
     * @return A {@link Single} emitting {@link ModelInfo}, or an error if the API is unreachable.
     */
    public Single<ModelInfo> fetchModelInfo(String modelName) {
        return modelDownloader.fetchModelInfo(modelName);
    }

    /**
     * Fetches a raw file from the model repository and returns it as parsed JSON.
     * Delegates to {@link VertxHuggingFaceClientRx#fetchFileAsJson(String, String)}.
     *
     * @param modelName  The HuggingFace model repository identifier.
     * @param filePath   Relative path within the repo, e.g. {@code "config.json"}.
     * @return A {@link Single} emitting the parsed {@link JsonObject}.
     */
    public Single<JsonObject> fetchFileAsJson(String modelName, String filePath) {
        return modelDownloader.fetchFileAsJson(modelName, filePath);
    }

    // --- private helpers ---

    private Single<String> downloadFile(String modelName, Path directory, String fileName) {
        Path outputPath = directory.resolve(fileName);

        return vertx
            .fileSystem()
            .rxExists(outputPath.toString())
            .flatMap(exists -> {
                if (exists) {
                    log.info("Skipping download; file already exists: {}", fileName);
                    return Single.just(outputPath.toString());
                }

                boolean isInSubDirectory = fileName.contains("/");

                return (isInSubDirectory ? buildSubDirectoryAndOpenFile(outputPath) : openFile(outputPath)).flatMap(file ->
                        modelDownloader
                            .downloadModelFile(modelName, fileName, file)
                            .doFinally(file::close)
                            .andThen(Single.just(outputPath.toString()))
                            .onErrorResumeNext(err ->
                                Single.error(
                                    new ModelDownloadFailedException(String.format("Download failed for model: %s", fileName), err)
                                )
                            )
                    );
            });
    }

    private Single<AsyncFile> buildSubDirectoryAndOpenFile(Path outputPath) {
        return vertx.fileSystem().mkdirs(outputPath.getParent().toString()).andThen(openFile(outputPath));
    }

    private Single<AsyncFile> openFile(Path outputPath) {
        return vertx.fileSystem().rxOpen(outputPath.toString(), new OpenOptions().setCreate(true).setWrite(true).setTruncateExisting(true));
    }
}
