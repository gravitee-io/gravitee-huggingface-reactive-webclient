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
import io.gravitee.reactive.webclient.api.ModelFileType;
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

    @Test
    @DisplayName("Should skip download when file exists locally")
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
    @DisplayName("Should download file when it does not exist locally")
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
    @DisplayName("Should download file when it does not exist locally and is in a subdirectory")
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
    @DisplayName("Should throw ModelFileNotFoundException when file is not available remotely")
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
    @DisplayName("Should throw ModelDownloadFailedException when download fails")
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
}
