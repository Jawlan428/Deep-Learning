package com.lftdetector;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;
import java.util.Map;

/**
 * Plain data holders for the JSON returned by detect.py.
 * Jackson maps these by field name.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class DetectionResult {

    public boolean ok;
    public String error;
    public String annotated_image;
    public String crop_image;
    public int image_width;
    public int image_height;
    public List<Detection> detections;
    public Classification classification;

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Detection {
        public int class_id;
        public String class_name;
        public double confidence;
        public double rotation_deg;
        public double bbox_width;
        public double bbox_height;

        // Per-detection Stage-2 results (null until "Read Result" runs). Each
        // detected cassette gets its own classification + cropped close-up so
        // the UI can review every result, not just the top one.
        public Classification classification;
        public String crop_image;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Classification {
        public String label;
        public double confidence;
        public Map<String, Double> probs;
        public String error;

        // Decision trace from the Stage-2 fusion layer (optional — null on the
        // older JSON shape). Lets the UI show WHY a verdict was reached.
        public String decision_source;   // e.g. "weak-positive rescue: 2 lines detected (CNN said negative)"
        public String classifier_backend; // "mobilenet_v3" / "yolov8-cls" / "none"
        public Integer num_lines;          // result lines counted by the classical detector
        public String cnn_label;           // what the neural classifier alone said
    }
}
