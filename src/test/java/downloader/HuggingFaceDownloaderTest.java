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
package downloader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import io.gravitee.reactive.webclient.api.FetchModelConfig;
import io.gravitee.reactive.webclient.api.ModelFile;
import io.gravitee.reactive.webclient.api.ModelFileInfo;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.gravitee.reactive.webclient.api.ModelInfo;
import io.gravitee.reactive.webclient.huggingface.client.VertxHuggingFaceClientRx;
import io.gravitee.reactive.webclient.huggingface.downloader.HuggingFaceDownloader;
import io.gravitee.reactive.webclient.huggingface.exception.ModelDownloadFailedException;
import io.gravitee.reactive.webclient.huggingface.exception.ModelFileNotFoundException;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.file.AsyncFile;
import io.vertx.rxjava3.core.file.FileSystem;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class HuggingFaceDownloaderTest {

    // -------------------------------------------------------------------------
    // fetchModel — existing behaviour
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("fetchModel: skips download when file exists locally")
    void shouldSkipDownloadWhenFileExistsLocally() {
        String modelName = "test-model";
        ModelFile file = new ModelFile("config.json", ModelFileType.CONFIG);
        Path modelDir = Path.of("temp-model-dir");

        var vertx = mock(Vertx.class);
        var fileSystem = mock(FileSystem.class);
        when(vertx.fileSystem()).thenReturn(fileSystem);
        when(vertx.fileSystem().mkdirs(any())).thenReturn(Completable.complete());
        when(fileSystem.rxExists(modelDir.resolve(file.name()).toString())).thenReturn(Single.just(true));

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.listModelFiles(modelName)).thenReturn(Flowable.just("config.json"));

        var service = new HuggingFaceDownloader(vertx, client);

        service
            .fetchModel(new FetchModelConfig(modelName, List.of(file), modelDir))
            .test()
            .awaitDone(5, TimeUnit.SECONDS)
            .assertComplete()
            .assertNoErrors()
            .assertValue(result -> {
                assertThat(result.get(ModelFileType.CONFIG)).contains("config.json");
                return true;
            });

        verify(client, never()).downloadModelFile(any(), any(), any());
    }

    @Test
    @DisplayName("fetchModel: downloads file when it does not exist locally")
    void shouldDownloadFileWhenNotExistsLocally() {
        String modelName = "test-model";
        ModelFile file = new ModelFile("config.json", ModelFileType.CONFIG);
        Path modelDir = Path.of("temp-model-dir");

        var vertx = mock(Vertx.class);
        var fileSystem = mock(FileSystem.class);
        var asyncFile = mock(AsyncFile.class);

        when(vertx.fileSystem()).thenReturn(fileSystem);
        when(vertx.fileSystem().mkdirs(any())).thenReturn(Completable.complete());
        when(fileSystem.rxExists(modelDir.resolve(file.name()).toString())).thenReturn(Single.just(false));
        when(fileSystem.rxOpen(eq(modelDir.resolve(file.name()).toString()), any())).thenReturn(Single.just(asyncFile));

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.listModelFiles(modelName)).thenReturn(Flowable.just(file.name()));
        when(client.downloadModelFile(modelName, file.name(), asyncFile)).thenReturn(Completable.complete());

        var fetcher = new HuggingFaceDownloader(vertx, client);

        fetcher
            .fetchModel(new FetchModelConfig(modelName, List.of(file), modelDir))
            .test()
            .awaitDone(5, TimeUnit.SECONDS)
            .assertComplete()
            .assertNoErrors()
            .assertValue(result -> {
                assertThat(result.get(ModelFileType.CONFIG)).contains("temp-model-dir/config.json");
                return true;
            });
        verify(client).downloadModelFile(modelName, file.name(), asyncFile);
        verify(asyncFile).close();
    }

    @Test
    @DisplayName("fetchModel: downloads file in subdirectory, creating parent dirs")
    void shouldDownloadFileWhenNotExistsLocallyAndWithinSubdirectory() {
        String modelName = "test-model";
        ModelFile file = new ModelFile("config/config.json", ModelFileType.CONFIG);
        Path modelDir = Path.of("temp-model-dir");

        var vertx = mock(Vertx.class);
        var fileSystem = mock(FileSystem.class);
        var asyncFile = mock(AsyncFile.class);

        when(vertx.fileSystem()).thenReturn(fileSystem);
        when(vertx.fileSystem().mkdirs(any())).thenReturn(Completable.complete());
        when(fileSystem.rxExists(modelDir.resolve(file.name()).toString())).thenReturn(Single.just(false));
        when(fileSystem.rxOpen(eq(modelDir.resolve(file.name()).toString()), any())).thenReturn(Single.just(asyncFile));

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.listModelFiles(modelName)).thenReturn(Flowable.just(file.name()));
        when(client.downloadModelFile(modelName, file.name(), asyncFile)).thenReturn(Completable.complete());

        var fetcher = new HuggingFaceDownloader(vertx, client);

        fetcher
            .fetchModel(new FetchModelConfig(modelName, List.of(file), modelDir))
            .test()
            .awaitDone(5, TimeUnit.SECONDS)
            .assertComplete()
            .assertNoErrors()
            .assertValue(result -> {
                assertThat(result.get(ModelFileType.CONFIG)).contains("temp-model-dir/config/config.json");
                return true;
            });
        verify(client).downloadModelFile(modelName, file.name(), asyncFile);
        verify(asyncFile).close();
    }

    @Test
    @DisplayName("fetchModel: throws ModelFileNotFoundException when file absent on remote")
    void shouldThrowWhenFileNotAvailableRemotely() {
        String modelName = "test-model";
        ModelFile file = new ModelFile("missing.json", ModelFileType.MODEL);
        Path modelDir = Path.of("temp-model-dir");

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.listModelFiles(modelName)).thenReturn(Flowable.just("other_file.json"));

        var fetcher = new HuggingFaceDownloader(mock(Vertx.class), client);

        fetcher.fetchModel(new FetchModelConfig(modelName, List.of(file), modelDir)).test().assertError(ModelFileNotFoundException.class);
    }

    @Test
    @DisplayName("fetchModel: throws ModelDownloadFailedException when download fails")
    void shouldThrowWhenDownloadFails() {
        String modelName = "test-model";
        ModelFile file = new ModelFile("model.onnx", ModelFileType.MODEL);
        Path modelDir = Path.of("temp-model-dir");

        var vertx = mock(Vertx.class);
        var fileSystem = mock(FileSystem.class);
        var asyncFile = mock(AsyncFile.class);

        when(vertx.fileSystem()).thenReturn(fileSystem);
        when(vertx.fileSystem().mkdirs(any())).thenReturn(Completable.complete());
        when(fileSystem.rxExists(any())).thenReturn(Single.just(false));
        when(fileSystem.rxOpen(any(), any())).thenReturn(Single.just(asyncFile));
        when(asyncFile.close()).thenReturn(Completable.complete());

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.listModelFiles(modelName)).thenReturn(Flowable.just("model.onnx"));
        when(client.downloadModelFile(modelName, "model.onnx", asyncFile))
            .thenReturn(Completable.error(new RuntimeException("network timeout")));

        var fetcher = new HuggingFaceDownloader(vertx, client);

        fetcher.fetchModel(new FetchModelConfig(modelName, List.of(file), modelDir)).test().assertError(ModelDownloadFailedException.class);
    }

    // -------------------------------------------------------------------------
    // fetchAllModelFiles — auto-discovery path
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("fetchAllModelFiles: downloads only files matching the predicate")
    void shouldFetchOnlyMatchingFiles() {
        String modelName = "bartowski/Llama-3.1-8B-GGUF";
        Path modelDir = Path.of("temp-model-dir");

        var modelInfo = new ModelInfo(
            modelName,
            false,
            false,
            List.of(
                new ModelFileInfo("Llama-3.1-8B-Q4_K_M.gguf", 4_000_000_000L),
                new ModelFileInfo("Llama-3.1-8B-Q8_0.gguf", 8_000_000_000L),
                new ModelFileInfo("README.md", 2048L)
            ),
            null
        );

        var vertx = mock(Vertx.class);
        var fileSystem = mock(FileSystem.class);
        var asyncFile = mock(AsyncFile.class);

        when(vertx.fileSystem()).thenReturn(fileSystem);
        when(fileSystem.rxExists(any())).thenReturn(Single.just(false));
        when(fileSystem.rxOpen(any(), any())).thenReturn(Single.just(asyncFile));
        when(asyncFile.close()).thenReturn(Completable.complete());

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.fetchModelInfo(modelName)).thenReturn(Single.just(modelInfo));
        when(client.downloadModelFile(eq(modelName), any(), eq(asyncFile))).thenReturn(Completable.complete());

        var fetcher = new HuggingFaceDownloader(vertx, client);

        // Only fetch Q4_K_M variant
        List<String> result = fetcher
            .fetchAllModelFiles(modelName, modelDir, name -> name.endsWith("Q4_K_M.gguf"))
            .test()
            .awaitDone(5, TimeUnit.SECONDS)
            .assertComplete()
            .assertNoErrors()
            .values()
            .get(0);

        assertThat(result).hasSize(1).allMatch(p -> p.contains("Q4_K_M.gguf"));
        verify(client).downloadModelFile(modelName, "Llama-3.1-8B-Q4_K_M.gguf", asyncFile);
        verify(client, never()).downloadModelFile(eq(modelName), eq("Llama-3.1-8B-Q8_0.gguf"), any());
        verify(client, never()).downloadModelFile(eq(modelName), eq("README.md"), any());
    }

    @Test
    @DisplayName("fetchAllModelFiles: emits empty list when no files match predicate")
    void shouldReturnEmptyListWhenNoFilesMatch() {
        String modelName = "some-org/some-model";
        Path modelDir = Path.of("temp-model-dir");

        var modelInfo = new ModelInfo(modelName, false, false, List.of(new ModelFileInfo("model.safetensors", 10_000_000L)), null);

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.fetchModelInfo(modelName)).thenReturn(Single.just(modelInfo));

        var fetcher = new HuggingFaceDownloader(mock(Vertx.class), client);

        fetcher
            .fetchAllModelFiles(modelName, modelDir, name -> name.endsWith(".gguf"))
            .test()
            .awaitDone(5, TimeUnit.SECONDS)
            .assertComplete()
            .assertNoErrors()
            .assertValue(List::isEmpty);
    }

    @Test
    @DisplayName("fetchAllModelFiles: skips files that already exist locally")
    void shouldSkipAlreadyDownloadedFilesInFetchAll() {
        String modelName = "bartowski/Llama-3.1-8B-GGUF";
        Path modelDir = Path.of("temp-model-dir");

        var modelInfo = new ModelInfo(
            modelName,
            false,
            false,
            List.of(new ModelFileInfo("Llama-3.1-8B-Q4_K_M.gguf", 4_000_000_000L)),
            null
        );

        var vertx = mock(Vertx.class);
        var fileSystem = mock(FileSystem.class);
        when(vertx.fileSystem()).thenReturn(fileSystem);
        when(fileSystem.rxExists(any())).thenReturn(Single.just(true)); // file already exists

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.fetchModelInfo(modelName)).thenReturn(Single.just(modelInfo));

        var fetcher = new HuggingFaceDownloader(vertx, client);

        fetcher
            .fetchAllModelFiles(modelName, modelDir, name -> name.endsWith(".gguf"))
            .test()
            .awaitDone(5, TimeUnit.SECONDS)
            .assertComplete()
            .assertNoErrors()
            .assertValue(result -> {
                assertThat(result).hasSize(1).allMatch(p -> p.contains("Q4_K_M.gguf"));
                return true;
            });

        verify(client, never()).downloadModelFile(any(), any(), any());
    }

    // -------------------------------------------------------------------------
    // fetchModelInfo delegation
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("fetchModelInfo: delegates to the underlying client")
    void shouldDelegateToClientForModelInfo() {
        String modelName = "meta-llama/Llama-3.1-8B";
        var modelInfo = new ModelInfo(modelName, true, false, List.of(), null);

        var client = mock(VertxHuggingFaceClientRx.class);
        when(client.fetchModelInfo(modelName)).thenReturn(Single.just(modelInfo));

        var fetcher = new HuggingFaceDownloader(mock(Vertx.class), client);

        ModelInfo result = fetcher.fetchModelInfo(modelName).blockingGet();

        assertThat(result).isEqualTo(modelInfo);
        verify(client).fetchModelInfo(modelName);
    }
}
