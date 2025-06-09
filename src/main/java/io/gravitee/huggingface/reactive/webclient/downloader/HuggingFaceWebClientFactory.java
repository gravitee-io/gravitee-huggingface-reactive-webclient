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

import io.vertx.ext.web.client.WebClientOptions;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.ext.web.client.WebClient;
import java.util.concurrent.TimeUnit;

public class HuggingFaceWebClientFactory {

    private static final int CONNECT_TIMEOUT = 5000;
    private static final int IDLE_TIMEOUT = 10000;
    private static final String HOST = "huggingface.co";

    public static WebClient createDefaultClient(Vertx vertx) {
        WebClientOptions clientOptions = new WebClientOptions()
            .setName("gio-hugging-face-client")
            .setDefaultHost(HOST)
            .setDefaultPort(443)
            .setSsl(true)
            .setConnectTimeout(CONNECT_TIMEOUT)
            .setIdleTimeout(IDLE_TIMEOUT)
            .setIdleTimeoutUnit(TimeUnit.MILLISECONDS)
            .setLogActivity(true);

        return WebClient.create(vertx, clientOptions);
    }
}
