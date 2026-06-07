package com.lftdetector;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.concurrent.Task;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * Runs normalize.py to produce a clean, EXIF-free version of an input image.
 * The result is what every other step (display, detection) operates on.
 */
public class NormalizationService {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String pythonExecutable;
    private final Path scriptPath;

    public NormalizationService(String pythonExecutable, Path scriptPath) {
        this.pythonExecutable = pythonExecutable;
        this.scriptPath = scriptPath;
    }

    public Task<Path> normalize(Path input, Path output) {
        return new Task<>() {
            @Override
            protected Path call() throws Exception {
                ProcessBuilder pb = new ProcessBuilder(
                        pythonExecutable,
                        scriptPath.toAbsolutePath().toString(),
                        input.toAbsolutePath().toString(),
                        output.toAbsolutePath().toString()
                );
                pb.redirectErrorStream(false);
                Process proc = pb.start();

                StringBuilder stdout = new StringBuilder();
                try (BufferedReader r = new BufferedReader(
                        new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        stdout.append(line).append('\n');
                    }
                }
                proc.waitFor();

                String out = stdout.toString().trim();
                if (out.isEmpty()) return null;

                String jsonLine = lastJsonLine(out);
                if (jsonLine == null) return null;

                JsonNode node = MAPPER.readTree(jsonLine);
                if (!node.path("ok").asBoolean(false)) return null;
                return Path.of(node.path("output").asText());
            }
        };
    }

    private static String lastJsonLine(String out) {
        String[] lines = out.split("\\R");
        for (int i = lines.length - 1; i >= 0; i--) {
            String t = lines[i].trim();
            if (t.startsWith("{") && t.endsWith("}")) return t;
        }
        return null;
    }
}