package com.lftdetector;

import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.concurrent.Task;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * Runs the Python detector in a background process and parses its JSON output.
 * Each detect() call returns a JavaFX Task you can attach handlers to.
 */
public class DetectionService {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String pythonExecutable;
    private final Path scriptPath;
    private final Path modelPath;

    public DetectionService(String pythonExecutable, Path scriptPath, Path modelPath) {
        this.pythonExecutable = pythonExecutable;
        this.scriptPath = scriptPath;
        this.modelPath = modelPath;
    }

    public Task<DetectionResult> detect(Path imagePath, Path outputImagePath) {
        return detect(imagePath, outputImagePath, null);
    }

    public Task<DetectionResult> detect(Path imagePath, Path outputImagePath, Path cropOutputPath) {
        return new Task<>() {
            @Override
            protected DetectionResult call() throws Exception {
                java.util.List<String> cmd = new java.util.ArrayList<>();
                cmd.add(pythonExecutable);
                cmd.add(scriptPath.toAbsolutePath().toString());
                cmd.add(modelPath.toAbsolutePath().toString());
                cmd.add(imagePath.toAbsolutePath().toString());
                cmd.add(outputImagePath.toAbsolutePath().toString());
                if (cropOutputPath != null) {
                    cmd.add(cropOutputPath.toAbsolutePath().toString());
                }
                ProcessBuilder pb = new ProcessBuilder(cmd);
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

                StringBuilder stderr = new StringBuilder();
                try (BufferedReader r = new BufferedReader(
                        new InputStreamReader(proc.getErrorStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        stderr.append(line).append('\n');
                    }
                }

                int code = proc.waitFor();
                String out = stdout.toString().trim();

                if (out.isEmpty()) {
                    DetectionResult err = new DetectionResult();
                    err.ok = false;
                    err.error = "Python produced no output (exit " + code + "). stderr: " + stderr;
                    return err;
                }

                // Python may print warnings before the JSON; grab the last line that parses.
                String jsonLine = lastJsonLine(out);
                if (jsonLine == null) {
                    DetectionResult err = new DetectionResult();
                    err.ok = false;
                    err.error = "No JSON found in Python output. Raw: " + out;
                    return err;
                }

                try {
                    return MAPPER.readValue(jsonLine, DetectionResult.class);
                } catch (IOException e) {
                    DetectionResult err = new DetectionResult();
                    err.ok = false;
                    err.error = "Failed to parse JSON: " + e.getMessage() + " | Line: " + jsonLine;
                    return err;
                }
            }
        };
    }

    private static String lastJsonLine(String out) {
        String[] lines = out.split("\\R");
        for (int i = lines.length - 1; i >= 0; i--) {
            String t = lines[i].trim();
            if (t.startsWith("{") && t.endsWith("}")) {
                return t;
            }
        }
        return null;
    }
}