package com.lftdetector;

import javafx.animation.Animation;
import javafx.animation.FadeTransition;
import javafx.animation.Interpolator;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.ParallelTransition;
import javafx.animation.PauseTransition;
import javafx.animation.ScaleTransition;
import javafx.animation.Timeline;
import javafx.animation.TranslateTransition;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.ScrollEvent;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.effect.GaussianBlur;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.SVGPath;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

public class LFTDetectorApp extends Application {

    private static final String PYTHON_EXECUTABLE = resolvePythonExecutable();
    private static final String DETECT_SCRIPT_REL  = "python/detect.py";
    private static final String NORMALIZE_SCRIPT_REL = "python/normalize.py";
    private static final String MODEL_PATH_REL = "best.pt";

    private static final String[] IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"};

    // Smooth decelerate — cubic-bezier(0.25, 0.46, 0.45, 0.94)
    private static final Interpolator EASE_OUT = Interpolator.SPLINE(0.25, 0.46, 0.45, 0.94);

    // Palette
    private static final String ACCENT         = "#00C49A";
    private static final String POSITIVE_COLOR = "#FF3B30";
    private static final String NEGATIVE_COLOR = "#30D158";
    private static final String INVALID_COLOR  = "#FF9500";

    private final List<Path> imagePaths = new ArrayList<>();
    private final Map<Path, Path> normalizedCache = new HashMap<>();
    private int currentIndex = -1;
    private Path tempDir;

    private StackPane rootContainer;
    private BorderPane mainLayout;
    private Stage primaryStage;

    private ImageView imageView;
    private StackPane imageStack;
    private Label noRoiOverlay;
    private Label folderLabel = new Label();
    private Label filenameLabel;
    private Label counterLabel;
    private Label statusLabel;
    private Button detectButton;
    private Button readResultButton;
    private Button prevButton;
    private Button nextButton;

    private VBox detailsContainer;

    private Path lastDetectedImage;
    private DetectionResult lastResult;
    private int selectedDetection = 0;
    private double imageZoom = 1.0;

    private VBox resultsPanel;
    private Timeline pulseTimeline;

    private DetectionService detectionService;
    private NormalizationService normalizationService;

    private HBox overlayLeftToolbar;
    private HBox overlayRightToolbar;
    private Timeline shimmerTimeline;
    private Button activeActionButton;

    // ==================== Bootstrap ====================

    @Override
    public void start(Stage stage) {
        this.primaryStage = stage;

        Path detectScript    = Paths.get(DETECT_SCRIPT_REL).toAbsolutePath();
        Path normalizeScript = Paths.get(NORMALIZE_SCRIPT_REL).toAbsolutePath();
        Path modelPath       = Paths.get(MODEL_PATH_REL).toAbsolutePath();
        detectionService     = new DetectionService(PYTHON_EXECUTABLE, detectScript, modelPath);
        normalizationService = new NormalizationService(PYTHON_EXECUTABLE, normalizeScript);

        try {
            tempDir = Files.createTempDirectory("lft-ui-");
            tempDir.toFile().deleteOnExit();
        } catch (IOException e) {
            tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "lft-ui");
            tempDir.toFile().mkdirs();
        }

        rootContainer = new StackPane();
        rootContainer.getStyleClass().add("root-pane");

        Scene scene = new Scene(rootContainer, 1180, 760);
        scene.getStylesheets().add(getClass().getResource("/app.css").toExternalForm());

        scene.addEventFilter(javafx.scene.input.KeyEvent.KEY_PRESSED, e -> {
            if (mainLayout != null && rootContainer.getChildren().contains(mainLayout)) {
                if (e.getCode() == KeyCode.LEFT)        { navigate(-1); e.consume(); }
                else if (e.getCode() == KeyCode.RIGHT)  { navigate(1);  e.consume(); }
                else if (e.getCode() == KeyCode.ENTER)  { runDetection(); e.consume(); }
            }
        });

        stage.setTitle("COVID-19 LFT Detector");
        stage.setScene(scene);
        stage.setMinWidth(980);
        stage.setMinHeight(640);
        stage.show();

        showWelcomePage();
    }

    // ==================== Page switching ====================

    private void showWelcomePage() {
        Node page = buildWelcomePage();
        page.setOpacity(0);
        page.setTranslateY(10);
        rootContainer.getChildren().setAll(page);
        new Timeline(new KeyFrame(Duration.millis(300),
            new KeyValue(page.opacityProperty(), 1.0, EASE_OUT),
            new KeyValue(page.translateYProperty(), 0, EASE_OUT)
        )).play();
    }

    private void showMainPage() {
        if (mainLayout == null) mainLayout = buildMainLayout();
        mainLayout.setOpacity(0);
        mainLayout.setTranslateY(8);
        rootContainer.getChildren().setAll(mainLayout);
        Timeline tl = new Timeline(new KeyFrame(Duration.millis(280),
            new KeyValue(mainLayout.opacityProperty(), 1.0, EASE_OUT),
            new KeyValue(mainLayout.translateYProperty(), 0, EASE_OUT)
        ));
        tl.setOnFinished(e -> animateToolbarEntrance());
        tl.play();
        updateUiState();
    }

    // ==================== Welcome Page ====================

    private Node buildWelcomePage() {
        // Ambient particle background layer
        Pane particles = buildParticleBackground();

        // Logo with gentle pulse
        StackPane logo = buildLogoCircle(60, 15);
        Timeline logoPulse = new Timeline(
            new KeyFrame(Duration.ZERO,
                new KeyValue(logo.scaleXProperty(), 1.0, EASE_OUT),
                new KeyValue(logo.scaleYProperty(), 1.0, EASE_OUT)),
            new KeyFrame(Duration.millis(2600),
                new KeyValue(logo.scaleXProperty(), 1.07, EASE_OUT),
                new KeyValue(logo.scaleYProperty(), 1.07, EASE_OUT)),
            new KeyFrame(Duration.millis(5200),
                new KeyValue(logo.scaleXProperty(), 1.0, EASE_OUT),
                new KeyValue(logo.scaleYProperty(), 1.0, EASE_OUT))
        );
        logoPulse.setCycleCount(Animation.INDEFINITE);
        logoPulse.play();

        Label appName = new Label("COVID-19 LFT Detector");
        appName.getStyleClass().add("welcome-title");

        // Elegant upload card
        VBox uploadCard = buildUploadCard();

        VBox content = new VBox();
        content.setAlignment(Pos.CENTER);
        content.setMaxWidth(500);
        VBox.setMargin(logo,       new Insets(0, 0, 28, 0));
        VBox.setMargin(appName,    new Insets(0, 0, 48, 0));
        VBox.setMargin(uploadCard, new Insets(0, 0, 0, 0));
        content.getChildren().addAll(logo, appName, uploadCard);

        StackPane page = new StackPane(particles, content);
        page.getStyleClass().add("welcome-page");
        return page;
    }

    /**
     * Three-layer particle field — Apple / VisionOS atmospheric style.
     *
     * Key design constraints for a LIGHT background (#F2F5FC):
     *   - Pure white particles on a white page = invisible. Must use chromatic tints.
     *   - Soft cool-blue / teal fills (low saturation) give contrast without garish color.
     *   - GaussianBlur spreads pixel energy: large blur on small circles kills visibility.
     *     Keep blur <= radius*0.40 on mid/small layers so shape centres stay legible.
     *   - Effective visible alpha = fill_alpha x node_opacity. Both must be high enough.
     *     Target effective alpha ~0.10–0.20 at rest so particles read clearly.
     *
     * Layer A  8 deep halos   — very large, heavily blurred, give page depth
     * Layer B  14 mid orbs    — the primary visible layer, gently breathing
     * Layer C  16 small dots  — crisp foreground sparkle, independent breathing
     */
    private Pane buildParticleBackground() {
        Pane pane = new Pane();
        pane.setMouseTransparent(true);
        pane.setMaxSize(Double.MAX_VALUE, Double.MAX_VALUE);

        Random rng = new Random(42);

        // ---- Layer A: deep background halos ----
        // Large radius, heavy blur, mid-alpha fill.
        // On a light page these read as soft pastel glows rather than blobs.
        double[][] halos = {
            {  90,  75, 110}, { 680,  50,  90}, {1120, 160, 105},
            {  45, 500,  88}, { 950, 410, 115}, { 400, 690,  95},
            {1080, 600,  92}, { 530, 330, 100}
        };
        for (double[] h : halos) {
            javafx.scene.shape.Circle c = new javafx.scene.shape.Circle(h[2]);
            c.setCenterX(h[0]); c.setCenterY(h[1]);
            // Soft blue-violet: visible on white without looking saturated
            c.setFill(Color.web("rgba(100,140,230,0.07)"));
            c.setEffect(new GaussianBlur(h[2] * 0.68));
            driftParticle(c, rng, 16000, 24000, 10, 22);
            pane.getChildren().add(c);
        }

        // ---- Layer B: mid-field orbs — main visible atmosphere ----
        // {cx, cy, radius, colorIndex}
        // 0=cool-blue  1=soft-teal  2=lavender
        double[][] orbs = {
            { 220, 150, 22, 0}, { 570, 105, 18, 1}, { 890,  60, 24, 0},
            {1155, 265, 20, 2}, { 150, 340, 26, 1}, { 730, 290, 17, 0},
            {1020, 450, 22, 2}, { 380, 490, 19, 1}, { 810, 570, 24, 0},
            {  50, 665, 18, 2}, { 610, 690, 22, 1}, {1120, 645, 20, 0},
            { 470, 250, 16, 2}, { 960, 200, 20, 1}
        };
        // Chromatic fills — distinguishable from the white/light-blue page background
        String[] orbFills = {
            "rgba(80,130,230,0.17)",    // cool blue
            "rgba(0,185,145,0.13)",     // soft teal (brand tint)
            "rgba(130,110,210,0.12)"    // lavender
        };
        for (double[] o : orbs) {
            javafx.scene.shape.Circle c = new javafx.scene.shape.Circle(o[2]);
            c.setCenterX(o[0]); c.setCenterY(o[1]);
            c.setFill(Color.web(orbFills[(int) o[3]]));
            // Light blur — keep the orb shape readable so it reads as a circle, not a smear
            c.setEffect(new GaussianBlur(o[2] * 0.38));
            driftParticle(c, rng, 9000, 15000, 20, 34);
            breatheOpacity(c, rng, 0.55, 1.0, 5000, 9000);
            pane.getChildren().add(c);
        }

        // ---- Layer C: small crisp dots — foreground sparkle ----
        // {cx, cy, radius}
        // Small radius + minimal blur = visible as distinct glowing points
        double[][] dots = {
            { 310,  72, 4.5}, { 750, 122, 3.5}, { 980, 188, 5.0},
            { 195, 238, 4.0}, { 585, 368, 4.5}, { 450, 152, 3.0},
            { 860, 312, 5.0}, {1110, 378, 3.5}, { 128, 438, 4.0},
            { 678, 528, 4.5}, {1000, 548, 3.5}, { 318, 612, 4.0},
            { 768, 652, 5.0}, {1048, 708, 4.0}, { 500, 600, 3.5},
            { 840, 720, 4.0}
        };
        for (double[] d : dots) {
            javafx.scene.shape.Circle c = new javafx.scene.shape.Circle(d[2]);
            c.setCenterX(d[0]); c.setCenterY(d[1]);
            c.setFill(Color.web("rgba(70,125,220,0.22)"));
            c.setEffect(new GaussianBlur(2.8));
            driftParticle(c, rng, 7000, 12000, 10, 18);
            breatheOpacity(c, rng, 0.45, 1.0, 3500, 7000);
            pane.getChildren().add(c);
        }

        return pane;
    }

    /**
     * Oscillates a particle between its origin and a random offset.
     * LINEAR interpolator + autoReverse produces a smooth sine-like triangle wave.
     * A random delay staggers startup so all particles don't move in unison.
     */
    private void driftParticle(javafx.scene.shape.Circle c, Random rng,
                                int minMs, int maxMs, double minPx, double maxPx) {
        double dur = minMs + rng.nextDouble() * (maxMs - minMs);
        double sign = rng.nextBoolean() ? 1 : -1;
        double dx = sign * (minPx + rng.nextDouble() * (maxPx - minPx));
        double dy = (rng.nextBoolean() ? 1 : -1) * (minPx + rng.nextDouble() * (maxPx - minPx));
        Timeline tl = new Timeline(
            new KeyFrame(Duration.ZERO,
                new KeyValue(c.translateXProperty(), 0,  Interpolator.LINEAR),
                new KeyValue(c.translateYProperty(), 0,  Interpolator.LINEAR)),
            new KeyFrame(Duration.millis(dur),
                new KeyValue(c.translateXProperty(), dx, Interpolator.LINEAR),
                new KeyValue(c.translateYProperty(), dy, Interpolator.LINEAR))
        );
        tl.setAutoReverse(true);
        tl.setCycleCount(Animation.INDEFINITE);
        tl.setDelay(Duration.millis(rng.nextInt(6000)));
        tl.play();
    }

    /**
     * Gentle opacity pulse -- makes particles feel alive rather than painted on.
     * Runs on a cycle independent of the drift so the two motions don't phase-lock.
     */
    private void breatheOpacity(javafx.scene.shape.Circle c, Random rng,
                                 double lo, double hi, int minMs, int maxMs) {
        double start = lo + rng.nextDouble() * (hi - lo);
        double dur   = minMs + rng.nextDouble() * (maxMs - minMs);
        Timeline tl = new Timeline(
            new KeyFrame(Duration.ZERO,
                new KeyValue(c.opacityProperty(), start, Interpolator.LINEAR)),
            new KeyFrame(Duration.millis(dur),
                new KeyValue(c.opacityProperty(), hi, Interpolator.LINEAR))
        );
        tl.setAutoReverse(true);
        tl.setCycleCount(Animation.INDEFINITE);
        tl.setDelay(Duration.millis(rng.nextInt(3000)));
        tl.play();
    }

    /** Floating glass upload card with animated folder icon and drag-drop support. */
    private VBox buildUploadCard() {
        // Folder icon (SVG path — simple open folder shape)
        Label folderIcon = new Label();
        folderIcon.getStyleClass().add("upload-folder-icon");
        folderIcon.setText("📂"); // 📂 open folder

        StackPane iconWrap = new StackPane(folderIcon);
        iconWrap.getStyleClass().add("upload-icon-wrap");
        iconWrap.setPrefSize(72, 72);
        iconWrap.setMaxSize(72, 72);

        Label uploadTitle = new Label("Choose a folder to begin analysis");
        uploadTitle.getStyleClass().add("upload-card-title");
        uploadTitle.setWrapText(true);
        uploadTitle.setAlignment(Pos.CENTER);
        uploadTitle.setTextAlignment(javafx.scene.text.TextAlignment.CENTER);

        Label uploadSub = new Label("Drag a folder here, or click Browse");
        uploadSub.getStyleClass().add("upload-card-sub");

        Button browseBtn = new Button("Browse Folder");
        browseBtn.getStyleClass().add("upload-browse-btn");
        browseBtn.setOnAction(e -> chooseFolderFromWelcome());
        addButtonAnimations(browseBtn);

        VBox card = new VBox();
        card.setAlignment(Pos.CENTER);
        card.getStyleClass().add("upload-card");
        VBox.setMargin(iconWrap,    new Insets(0, 0, 20, 0));
        VBox.setMargin(uploadTitle, new Insets(0, 0, 8, 0));
        VBox.setMargin(uploadSub,   new Insets(0, 0, 28, 0));
        card.getChildren().addAll(iconWrap, uploadTitle, uploadSub, browseBtn);

        // Card hover: very subtle lift
        card.setOnMouseEntered(e -> new Timeline(new KeyFrame(Duration.millis(160),
            new KeyValue(card.scaleXProperty(), 1.012, EASE_OUT),
            new KeyValue(card.scaleYProperty(), 1.012, EASE_OUT)
        )).play());
        card.setOnMouseExited(e -> new Timeline(new KeyFrame(Duration.millis(200),
            new KeyValue(card.scaleXProperty(), 1.0, EASE_OUT),
            new KeyValue(card.scaleYProperty(), 1.0, EASE_OUT)
        )).play());

        // Folder icon bounce on hover
        card.setOnMouseMoved(e -> { /* no-op — icon responds to card entered/exited */ });
        iconWrap.setOnMouseEntered(e -> new Timeline(new KeyFrame(Duration.millis(130),
            new KeyValue(folderIcon.scaleXProperty(), 1.18, EASE_OUT),
            new KeyValue(folderIcon.scaleYProperty(), 1.18, EASE_OUT)
        )).play());
        iconWrap.setOnMouseExited(e -> new Timeline(new KeyFrame(Duration.millis(190),
            new KeyValue(folderIcon.scaleXProperty(), 1.0, EASE_OUT),
            new KeyValue(folderIcon.scaleYProperty(), 1.0, EASE_OUT)
        )).play());

        // Drag-and-drop
        card.setOnDragOver(e -> {
            if (e.getDragboard().hasFiles()) e.acceptTransferModes(TransferMode.COPY);
            e.consume();
        });
        card.setOnDragDropped(e -> {
            List<File> files = e.getDragboard().getFiles();
            if (!files.isEmpty()) {
                File f = files.get(0);
                Path folder = f.isDirectory() ? f.toPath() : f.toPath().getParent();
                if (folder != null) { showMainPage(); loadFolder(folder); }
            }
            e.setDropCompleted(true);
            e.consume();
        });
        card.setOnDragEntered(e -> card.getStyleClass().add("upload-card-drag"));
        card.setOnDragExited(e  -> card.getStyleClass().remove("upload-card-drag"));

        return card;
    }

    private void chooseFolderFromWelcome() {
        DirectoryChooser dc = new DirectoryChooser();
        dc.setTitle("Choose folder of test images");
        File chosen = dc.showDialog(primaryStage);
        if (chosen == null) return;
        showMainPage();
        loadFolder(chosen.toPath());
    }

    // ==================== Main Layout ====================

    private BorderPane buildMainLayout() {
        BorderPane root = new BorderPane();
        root.setCenter(buildCenter());
        root.setBottom(buildBottomBar());
        return root;
    }

    // ==================== Header (unused but kept for safety) ====================

    private HBox buildHeader() {
        Button backBtn = new Button("<- Home");
        backBtn.getStyleClass().add("back-button");
        backBtn.setOnAction(e -> showWelcomePage());
        HBox header = new HBox(backBtn);
        header.setAlignment(Pos.CENTER_LEFT);
        header.getStyleClass().add("app-header");
        return header;
    }

    /** Virus logo circle at a given outer size and padding. */
    private StackPane buildLogoCircle(double size, double pad) {
        StackPane logo = new StackPane();
        logo.getStyleClass().add("logo-circle");
        logo.setPrefSize(size, size);
        logo.setMinSize(size, size);
        logo.setMaxSize(size, size);

        javafx.scene.Group virus = new javafx.scene.Group();
        double scale = (size - pad * 2) / 22.0;
        double coreR = 5 * scale, spikeInner = 7.5 * scale, spikeOuter = 11 * scale, tipR = 1.6 * scale;
        javafx.scene.shape.Circle core = new javafx.scene.shape.Circle(0, 0, coreR);
        core.setFill(Color.web(ACCENT));
        virus.getChildren().add(core);
        for (int i = 0; i < 8; i++) {
            double angle = 2 * Math.PI * i / 8;
            double x1 = Math.cos(angle) * spikeInner, y1 = Math.sin(angle) * spikeInner;
            double x2 = Math.cos(angle) * spikeOuter, y2 = Math.sin(angle) * spikeOuter;
            javafx.scene.shape.Line spike = new javafx.scene.shape.Line(x1, y1, x2, y2);
            spike.setStroke(Color.web(ACCENT));
            spike.setStrokeWidth(Math.max(1.0, 1.6 * scale));
            spike.setStrokeLineCap(javafx.scene.shape.StrokeLineCap.ROUND);
            virus.getChildren().add(spike);
            javafx.scene.shape.Circle tip = new javafx.scene.shape.Circle(x2, y2, tipR);
            tip.setFill(Color.web(ACCENT));
            virus.getChildren().add(tip);
        }
        logo.getChildren().add(virus);
        return logo;
    }

    private Label separatorDot() {
        Label l = new Label("|");
        l.getStyleClass().add("sep-dot");
        return l;
    }

    // ==================== Center ====================

    private HBox buildCenter() {
        // ---- Hero image viewer ----
        imageView = new ImageView();
        imageView.setPreserveRatio(true);

        imageStack = new StackPane(imageView);
        imageStack.getStyleClass().add("image-stack");
        imageStack.setMinSize(400, 400);

        imageView.fitWidthProperty().bind(imageStack.widthProperty());
        imageView.fitHeightProperty().bind(imageStack.heightProperty());

        Rectangle clip = new Rectangle();
        clip.widthProperty().bind(imageStack.widthProperty());
        clip.heightProperty().bind(imageStack.heightProperty());
        clip.setArcWidth(28); clip.setArcHeight(28);
        imageStack.setClip(clip);

        imageStack.setOnScroll((ScrollEvent e) -> {
            if (imageView.getImage() == null) return;
            double factor = e.getDeltaY() > 0 ? 1.1 : 1.0 / 1.1;
            imageZoom = Math.max(1.0, Math.min(5.0, imageZoom * factor));
            imageView.setScaleX(imageZoom);
            imageView.setScaleY(imageZoom);
            e.consume();
        });
        imageStack.setOnMouseClicked(e -> { if (e.getClickCount() == 2) resetZoom(); });

        // ---- Light segmented control (top-center) ----
        overlayLeftToolbar = buildSegmentedToolbar();
        overlayLeftToolbar.setMaxWidth(Region.USE_PREF_SIZE);
        overlayLeftToolbar.setMaxHeight(Region.USE_PREF_SIZE);
        StackPane.setAlignment(overlayLeftToolbar, Pos.TOP_CENTER);
        StackPane.setMargin(overlayLeftToolbar, new Insets(14, 0, 0, 0));

        // ---- Overlays ----
        noRoiOverlay = new Label("NO ROI");
        noRoiOverlay.getStyleClass().add("no-roi-label");
        noRoiOverlay.setVisible(false);
        StackPane.setAlignment(noRoiOverlay, Pos.CENTER);

        filenameLabel = new Label("--");
        filenameLabel.getStyleClass().add("image-chip");
        StackPane.setAlignment(filenameLabel, Pos.BOTTOM_LEFT);
        StackPane.setMargin(filenameLabel, new Insets(0, 0, 14, 14));

        imageStack.getChildren().addAll(noRoiOverlay, overlayLeftToolbar, filenameLabel);

        // ---- Analysis panel ----
        VBox rightPanel = buildResultsPanel();
        rightPanel.setPrefWidth(320);
        rightPanel.setMinWidth(280);
        rightPanel.setMaxWidth(340);

        HBox center = new HBox(10, imageStack, rightPanel);
        center.setPadding(new Insets(10, 10, 10, 10));
        HBox.setHgrow(imageStack, Priority.ALWAYS);
        return center;
    }

    /**
     * Light frosted-glass segmented control: Home | Detect | Read Results.
     * Replaces the previous dark pill toolbars.
     */
    private HBox buildSegmentedToolbar() {
        Button homeBtn = new Button("Home");
        homeBtn.getStyleClass().addAll("seg-btn", "seg-btn-home");
        homeBtn.setOnAction(e -> showWelcomePage());
        addButtonAnimations(homeBtn);

        detectButton = new Button("Detect");
        detectButton.getStyleClass().addAll("seg-btn", "seg-btn-detect");
        detectButton.setOnAction(e -> runDetection());
        addButtonAnimations(detectButton);

        readResultButton = new Button("Read Results");
        readResultButton.getStyleClass().addAll("seg-btn", "seg-btn-read");
        readResultButton.setOnAction(e -> runClassification());
        addButtonAnimations(readResultButton);

        Region sep1 = new Region(); sep1.getStyleClass().add("seg-sep");
        Region sep2 = new Region(); sep2.getStyleClass().add("seg-sep");

        HBox bar = new HBox(0, homeBtn, sep1, detectButton, sep2, readResultButton);
        bar.setAlignment(Pos.CENTER);
        bar.getStyleClass().add("segmented-control");
        return bar;
    }

    private VBox buildResultsPanel() {
        Label heading = new Label("Analysis");
        heading.getStyleClass().add("panel-title");

        detailsContainer = new VBox(16);
        detailsContainer.getStyleClass().add("details-panel");
        renderEmpty("Press Detect to locate the test cassette.");

        ScrollPane scroll = new ScrollPane(detailsContainer);
        scroll.setFitToWidth(true);
        scroll.getStyleClass().add("details-scroll");
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        VBox.setVgrow(scroll, Priority.ALWAYS);

        resultsPanel = new VBox(14, heading, scroll);
        resultsPanel.getStyleClass().add("results-panel");
        resultsPanel.setPrefWidth(370);
        resultsPanel.setMinWidth(330);
        return resultsPanel;
    }

    // ==================== Bottom navigation ====================

    private HBox buildBottomBar() {
        prevButton = new Button("← Previous");
        nextButton = new Button("Next →");
        prevButton.getStyleClass().add("nav-button");
        nextButton.getStyleClass().add("nav-button");
        prevButton.setMinWidth(140);
        nextButton.setMinWidth(140);
        prevButton.setOnAction(e -> navigate(-1));
        nextButton.setOnAction(e -> navigate(1));
        addButtonAnimations(prevButton);
        addButtonAnimations(nextButton);

        counterLabel = new Label("No images");
        counterLabel.getStyleClass().add("image-counter");
        statusLabel = new Label("Choose a folder to begin");
        statusLabel.getStyleClass().add("status-label");

        VBox centerBox = new VBox(3, counterLabel, statusLabel);
        centerBox.setAlignment(Pos.CENTER);

        Region spaceL = new Region();
        Region spaceR = new Region();
        HBox.setHgrow(spaceL, Priority.ALWAYS);
        HBox.setHgrow(spaceR, Priority.ALWAYS);

        HBox bar = new HBox(16, prevButton, spaceL, centerBox, spaceR, nextButton);
        bar.setAlignment(Pos.CENTER);
        bar.getStyleClass().add("bottom-bar");
        return bar;
    }

    // ==================== Folder + navigation ====================

    private void chooseFolder() {
        DirectoryChooser dc = new DirectoryChooser();
        dc.setTitle("Choose folder of test images");
        File chosen = dc.showDialog(primaryStage);
        if (chosen == null) return;
        loadFolder(chosen.toPath());
    }

    private void loadFolder(Path folder) {
        imagePaths.clear();
        normalizedCache.clear();
        try (Stream<Path> stream = Files.list(folder)) {
            stream.filter(this::isImage)
                    .sorted(Comparator.comparing(p -> p.getFileName().toString().toLowerCase(Locale.ROOT)))
                    .forEach(imagePaths::add);
        } catch (Exception e) {
            showAlert("Failed to read folder: " + e.getMessage());
            return;
        }

        folderLabel.setText(shortFolderName(folder));

        if (imagePaths.isEmpty()) {
            currentIndex = -1;
            statusLabel.setText("No images found in folder");
        } else {
            currentIndex = 0;
            statusLabel.setText("Loaded " + imagePaths.size() + " images");
        }
        showCurrent();
    }

    private boolean isImage(Path p) {
        if (!Files.isRegularFile(p)) return false;
        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String ext : IMAGE_EXTS) if (name.endsWith(ext)) return true;
        return false;
    }

    private String shortFolderName(Path folder) {
        Path parent = folder.getParent();
        if (parent == null) return folder.toString();
        return parent.getFileName() + "/" + folder.getFileName() + "/";
    }

    private void navigate(int delta) {
        if (imagePaths.isEmpty()) return;
        currentIndex = Math.floorMod(currentIndex + delta, imagePaths.size());
        showCurrent();
    }

    private void showCurrent() {
        lastResult = null;
        lastDetectedImage = null;
        selectedDetection = 0;
        updateImageBorder(null);
        renderEmpty("Press Detect to locate the test cassette.");

        if (currentIndex < 0 || imagePaths.isEmpty()) {
            clearViewerImage();
            filenameLabel.setText("--");
            counterLabel.setText("No images");
            updateUiState();
            return;
        }
        Path original = imagePaths.get(currentIndex);
        filenameLabel.setText(original.getFileName().toString());
        counterLabel.setText("Image " + (currentIndex + 1) + " of " + imagePaths.size());
        updateUiState();

        Path cached = normalizedCache.get(original);
        if (cached != null && Files.exists(cached)) {
            setViewerImage(cached);
            statusLabel.setText("Ready — " + imagePaths.size() + " images loaded");
            prefetchNeighbours();
            return;
        }

        statusLabel.setText("Loading...");
        int capturedIndex = currentIndex;
        ensureNormalized(original, () -> {
            if (capturedIndex == currentIndex) {
                Path n = normalizedCache.get(original);
                if (n != null && Files.exists(n)) setViewerImage(n);
                statusLabel.setText("Ready — " + imagePaths.size() + " images loaded");
                prefetchNeighbours();
            }
        });
    }

    private void ensureNormalized(Path original, Runnable onDone) {
        if (normalizedCache.containsKey(original)) {
            if (onDone != null) Platform.runLater(onDone);
            return;
        }
        Path normOut = tempDir.resolve("norm_" + System.currentTimeMillis() + "_"
                + safeName(original.getFileName().toString()));
        var task = normalizationService.normalize(original, normOut);
        task.setOnSucceeded(ev -> {
            Path result = task.getValue();
            if (result != null && Files.exists(result)) normalizedCache.put(original, result);
            if (onDone != null) onDone.run();
        });
        task.setOnFailed(ev -> { if (onDone != null) onDone.run(); });
        Thread t = new Thread(task, "normalize-task");
        t.setDaemon(true);
        t.start();
    }

    private void prefetchNeighbours() {
        if (imagePaths.size() < 2) return;
        int next = Math.floorMod(currentIndex + 1, imagePaths.size());
        int prev = Math.floorMod(currentIndex - 1, imagePaths.size());
        if (next != currentIndex) ensureNormalized(imagePaths.get(next), null);
        if (prev != currentIndex && prev != next) ensureNormalized(imagePaths.get(prev), null);
    }

    private static String safeName(String name) {
        String base = name.toLowerCase(Locale.ROOT);
        int dot = base.lastIndexOf('.');
        if (dot >= 0) base = base.substring(0, dot);
        base = base.replaceAll("[^a-z0-9]+", "_");
        if (base.length() > 40) base = base.substring(0, 40);
        return base + ".jpg";
    }

    private void updateUiState() {
        boolean hasImage = currentIndex >= 0 && !imagePaths.isEmpty();
        detectButton.setDisable(!hasImage);
        prevButton.setDisable(!hasImage);
        nextButton.setDisable(!hasImage);
        boolean hasDetections = lastResult != null && lastResult.detections != null
                && !lastResult.detections.isEmpty();
        readResultButton.setDisable(!hasDetections);
    }

    // ==================== Viewer helpers ====================

    private void setViewerImage(Path p) {
        resetZoom();
        Image newImage = new Image(p.toUri().toString(), false);
        if (imageView.getImage() != null) {
            Timeline out = new Timeline(new KeyFrame(Duration.millis(130),
                new KeyValue(imageView.opacityProperty(), 0.0, EASE_OUT),
                new KeyValue(imageView.scaleXProperty(), 0.96, EASE_OUT),
                new KeyValue(imageView.scaleYProperty(), 0.96, EASE_OUT)
            ));
            out.setOnFinished(ev -> {
                imageView.setImage(newImage);
                imageView.setScaleX(1.04);
                imageView.setScaleY(1.04);
                new Timeline(new KeyFrame(Duration.millis(200),
                    new KeyValue(imageView.opacityProperty(), 1.0, EASE_OUT),
                    new KeyValue(imageView.scaleXProperty(), 1.0, EASE_OUT),
                    new KeyValue(imageView.scaleYProperty(), 1.0, EASE_OUT)
                )).play();
            });
            out.play();
        } else {
            imageView.setImage(newImage);
            imageView.setOpacity(0);
            imageView.setScaleX(1.04);
            imageView.setScaleY(1.04);
            new Timeline(new KeyFrame(Duration.millis(220),
                new KeyValue(imageView.opacityProperty(), 1.0, EASE_OUT),
                new KeyValue(imageView.scaleXProperty(), 1.0, EASE_OUT),
                new KeyValue(imageView.scaleYProperty(), 1.0, EASE_OUT)
            )).play();
        }
    }

    private void clearViewerImage() {
        resetZoom();
        imageView.setImage(null);
    }

    private void resetZoom() {
        imageZoom = 1.0;
        if (imageView != null) { imageView.setScaleX(1.0); imageView.setScaleY(1.0); }
    }

    // ==================== Detection (Stage 1) ====================

    private void runDetection() {
        if (currentIndex < 0 || imagePaths.isEmpty()) return;
        Path original  = imagePaths.get(currentIndex);
        Path normalized = normalizedCache.get(original);
        if (normalized == null || !Files.exists(normalized)) {
            statusLabel.setText("Image not ready yet — wait a moment");
            return;
        }

        Path outImg = tempDir.resolve("annotated_" + System.currentTimeMillis() + ".jpg");

        lastResult = null;
        lastDetectedImage = null;
        selectedDetection = 0;
        statusLabel.setText("Detecting...");
        activeActionButton = detectButton;
        setProcessing(true);

        var task = detectionService.detect(normalized, outImg);
        task.setOnSucceeded(ev -> {
            DetectionResult result = task.getValue();
            if (!result.ok) {
                statusLabel.setText("Detection failed");
                renderError(result.error);
            } else if (result.detections == null || result.detections.isEmpty()) {
                statusLabel.setText("No cassette detected");
                updateImageBorder("noroi");
                renderNoRoi();
            } else {
                lastResult = result;
                lastDetectedImage = normalized;
                selectedDetection = 0;
                statusLabel.setText("Detected " + result.detections.size()
                        + " cassette" + (result.detections.size() == 1 ? "" : "s"));
                if (result.annotated_image != null) {
                    Path annotated = Paths.get(result.annotated_image);
                    if (Files.exists(annotated)) setViewerImage(annotated);
                }
                renderResult();
            }
            setProcessing(false);
        });
        task.setOnFailed(ev -> {
            Throwable err = task.getException();
            statusLabel.setText("Detection error");
            renderError(err == null ? "unknown" : err.getMessage());
            setProcessing(false);
        });

        Thread t = new Thread(task, "detection-task");
        t.setDaemon(true);
        t.start();
    }

    // ==================== Classification (Stage 2) ====================

    private void runClassification() {
        if (lastDetectedImage == null || !Files.exists(lastDetectedImage)) {
            statusLabel.setText("Run Detect first");
            return;
        }

        Path outImg  = tempDir.resolve("annotated2_" + System.currentTimeMillis() + ".jpg");
        Path cropImg = tempDir.resolve("crop_"       + System.currentTimeMillis() + ".jpg");

        statusLabel.setText("Reading results...");
        activeActionButton = readResultButton;
        setProcessing(true);

        var task = detectionService.detect(lastDetectedImage, outImg, cropImg);
        task.setOnSucceeded(ev -> {
            DetectionResult result = task.getValue();
            if (!result.ok) {
                statusLabel.setText("Classification failed");
                renderError(result.error);
            } else {
                lastResult = result;
                int n = result.detections == null ? 0 : result.detections.size();
                if (selectedDetection < 0 || selectedDetection >= n) selectedDetection = 0;
                statusLabel.setText(summaryStatus(result));
                if (n > 0) {
                    DetectionResult.Detection d = result.detections.get(selectedDetection);
                    if (d.crop_image != null) {
                        Path crop = Paths.get(d.crop_image);
                        if (Files.exists(crop)) setViewerImage(crop);
                    } else if (result.annotated_image != null) {
                        Path ann = Paths.get(result.annotated_image);
                        if (Files.exists(ann)) setViewerImage(ann);
                    }
                }
                renderResult();
            }
            setProcessing(false);
        });
        task.setOnFailed(ev -> {
            Throwable err = task.getException();
            statusLabel.setText("Analysis error");
            renderError(err == null ? "unknown" : err.getMessage());
            setProcessing(false);
        });

        Thread t = new Thread(task, "classify-task");
        t.setDaemon(true);
        t.start();
    }

    private String summaryStatus(DetectionResult result) {
        if (result.detections == null || result.detections.isEmpty()) return "No results";
        if (result.detections.size() == 1) {
            DetectionResult.Classification c = result.detections.get(0).classification;
            if (c != null && c.error == null && c.label != null)
                return "Result: " + c.label.toUpperCase(Locale.ROOT);
            return "Result ready";
        }
        return result.detections.size() + " results ready";
    }

    // ==================== Results rendering ====================

    private void animateIn(Node node) {
        node.setOpacity(0);
        node.setTranslateY(10);
        node.setScaleX(0.988);
        node.setScaleY(0.988);
        new Timeline(new KeyFrame(Duration.millis(240),
            new KeyValue(node.opacityProperty(),    1.0, EASE_OUT),
            new KeyValue(node.translateYProperty(), 0,   EASE_OUT),
            new KeyValue(node.scaleXProperty(),     1.0, EASE_OUT),
            new KeyValue(node.scaleYProperty(),     1.0, EASE_OUT)
        )).play();
    }

    private void renderEmpty(String message) {
        if (detailsContainer == null) return;
        Label hint = new Label(message);
        hint.getStyleClass().add("hint-text");
        hint.setWrapText(true);
        VBox box = new VBox(hint);
        box.getStyleClass().add("placeholder-card");
        detailsContainer.getChildren().setAll(box);
        animateIn(box);
    }

    private void renderError(String message) {
        if (detailsContainer == null) return;
        Label heading = new Label("Something went wrong");
        heading.getStyleClass().add("error-title");
        Label detail = new Label(message == null ? "An unexpected error occurred." : message);
        detail.getStyleClass().add("hint-text");
        detail.setWrapText(true);
        VBox box = new VBox(8, heading, detail);
        box.getStyleClass().add("placeholder-card");
        detailsContainer.getChildren().setAll(box);
        animateIn(box);
    }

    private void renderNoRoi() {
        if (detailsContainer == null) return;
        Label tag = new Label("No Cassette Found");
        tag.getStyleClass().add("no-roi-tag");
        Label sub = new Label("No test cassette was found in this image. Try a clearer photo.");
        sub.getStyleClass().add("no-roi-sub");
        sub.setWrapText(true);
        VBox box = new VBox(10, tag, sub);
        box.getStyleClass().add("no-roi-card");
        box.setAlignment(Pos.CENTER);
        detailsContainer.getChildren().setAll(box);
        animateIn(box);
    }

    private void renderResult() {
        if (detailsContainer == null) return;
        detailsContainer.getChildren().clear();
        if (lastResult == null || lastResult.detections == null || lastResult.detections.isEmpty()) {
            renderEmpty("Press Detect to locate the test cassette.");
            return;
        }
        List<DetectionResult.Detection> dets = lastResult.detections;
        if (selectedDetection < 0 || selectedDetection >= dets.size()) selectedDetection = 0;

        DetectionResult.Detection sel = dets.get(selectedDetection);
        String selLabel = (sel.classification != null && sel.classification.error == null)
                ? sel.classification.label : null;
        updateImageBorder(selLabel);

        Node summaryCard = buildSummaryCard(dets);
        detailsContainer.getChildren().add(summaryCard);
        animateIn(summaryCard);

        for (int i = 0; i < dets.size(); i++) {
            Node card = buildDetectionCard(dets.get(i), i);
            detailsContainer.getChildren().add(card);
            final int delay = 80 + i * 55;
            Platform.runLater(() -> {
                PauseTransition pause = new PauseTransition(Duration.millis(delay));
                pause.setOnFinished(e -> animateIn(card));
                pause.play();
            });
        }
    }

    private VBox buildSummaryCard(List<DetectionResult.Detection> dets) {
        Label heading = new Label("SUMMARY");
        heading.getStyleClass().add("panel-heading");

        VBox card = new VBox(10, heading);
        card.getStyleClass().add("summary-card");

        Label count = new Label(dets.size() + (dets.size() == 1 ? " cassette detected" : " cassettes detected"));
        count.getStyleClass().add("summary-count");
        card.getChildren().add(count);

        boolean anyClassified = dets.stream().anyMatch(d -> d.classification != null);
        if (!anyClassified) {
            Label note = new Label("Press Read Results to classify each cassette.");
            note.getStyleClass().add("hint-text");
            note.setWrapText(true);
            card.getChildren().add(note);
            return card;
        }

        int pos = 0, neg = 0, inv = 0, unk = 0;
        for (DetectionResult.Detection d : dets) {
            String label = (d.classification != null && d.classification.error == null)
                    ? d.classification.label : null;
            if (label == null) { unk++; continue; }
            switch (label.toLowerCase(Locale.ROOT)) {
                case "positive": pos++; break;
                case "negative": neg++; break;
                case "invalid":  inv++; break;
                default: unk++;
            }
        }
        FlowPane tallies = new FlowPane(12, 6);
        if (pos > 0) tallies.getChildren().add(tally(pos, "Positive", POSITIVE_COLOR));
        if (neg > 0) tallies.getChildren().add(tally(neg, "Negative", NEGATIVE_COLOR));
        if (inv > 0) tallies.getChildren().add(tally(inv, "Invalid",  INVALID_COLOR));
        if (unk > 0) tallies.getChildren().add(tally(unk, "Unread",   "#AEAEB2"));
        card.getChildren().add(tallies);
        return card;
    }

    private HBox tally(int n, String label, String color) {
        Label dot  = new Label("●");
        dot.setStyle("-fx-text-fill: " + color + "; -fx-font-size: 9;");
        Label text = new Label(n + " " + label);
        text.getStyleClass().add("tally-text");
        HBox row = new HBox(6, dot, text);
        row.setAlignment(Pos.CENTER_LEFT);
        return row;
    }

    private VBox buildDetectionCard(DetectionResult.Detection d, int idx) {
        boolean selected = (idx == selectedDetection);
        DetectionResult.Classification c = d.classification;

        Label title = new Label("Detection " + (idx + 1));
        title.getStyleClass().add("det-title");
        Region grow = new Region();
        HBox.setHgrow(grow, Priority.ALWAYS);
        HBox header = new HBox(8, title, grow);
        header.setAlignment(Pos.CENTER_LEFT);
        if (c != null && c.error == null && c.label != null) header.getChildren().add(badge(c.label));
        else if (c != null && c.error != null) header.getChildren().add(badge("error"));

        VBox card = new VBox(12, header);
        card.getStyleClass().add("detection-card");

        if (selected) {
            String resultLabel = (c != null && c.error == null && c.label != null) ? c.label : null;
            String bgColor   = resultLabel != null ? verdictBg(resultLabel)    : "rgba(0,196,154,0.05)";
            String bdrColor  = resultLabel != null ? verdictColor(resultLabel) : ACCENT;
            String glowColor = resultLabel != null ? verdictGlow(resultLabel)  : "rgba(0,196,154,0.14)";
            card.setStyle(String.format(
                "-fx-background-color: %s; -fx-border-color: %s; -fx-border-width: 1.5; " +
                "-fx-border-radius: 16; -fx-effect: dropshadow(gaussian, %s, 22, 0, 0, 0);",
                bgColor, bdrColor, glowColor));
        }
        card.setOnMouseClicked(e -> selectDetection(idx));

        if (d.crop_image != null) {
            Path crop = Paths.get(d.crop_image);
            if (Files.exists(crop)) {
                ImageView thumb = new ImageView(new Image(crop.toUri().toString(), false));
                thumb.setPreserveRatio(true);
                thumb.setSmooth(true);
                thumb.setFitWidth(selected  ? 316 : 200);
                thumb.setFitHeight(selected ? 180 : 110);
                StackPane frame = new StackPane(thumb);
                frame.getStyleClass().add("thumb-frame");
                // Soft glow on the cassette crop only — no color anywhere else
                if (selected && c != null && c.error == null && c.label != null) {
                    frame.setStyle(String.format(
                        "-fx-effect: dropshadow(gaussian, %s, 30, 0.12, 0, 0);",
                        verdictColor(c.label)));
                }
                card.getChildren().add(frame);
            }
        }

        card.getChildren().add(compactMetric("Confidence",
                String.format(Locale.ROOT, "%.0f%%", d.confidence * 100)));

        if (selected) {
            card.getChildren().add(buildSelectedDetail(d, c));
        } else if (c == null) {
            Label hint = new Label("Not yet classified");
            hint.getStyleClass().add("muted-text");
            card.getChildren().add(hint);
        }
        return card;
    }

    private VBox buildSelectedDetail(DetectionResult.Detection d, DetectionResult.Classification c) {
        VBox box = new VBox(8);
        List<Node> sections = new ArrayList<>();

        if (c != null && c.error == null && c.label != null) {
            // Animated confidence counter
            sections.add(buildAnimatedConfidenceLabel(c.confidence));

            // Class probability bars
            if (c.probs != null) {
                VBox probsSection = new VBox(6);
                Label probHeading = new Label("AI CONFIDENCE DETAILS");
                probHeading.getStyleClass().add("metric-label");
                VBox.setMargin(probHeading, new Insets(6, 0, 2, 0));
                probsSection.getChildren().add(probHeading);
                for (String cls : new String[]{"positive", "negative", "invalid"}) {
                    Double p = c.probs.get(cls);
                    if (p == null) continue;
                    probsSection.getChildren().add(probBar(cls, p));
                }
                sections.add(probsSection);
            }

            // Lines detected (clean — no developer strings)
            if (c.num_lines != null) {
                VBox lineSection = new VBox(2);
                Label lbl = new Label("Lines detected: " + c.num_lines);
                lbl.getStyleClass().add("hint-text");
                VBox.setMargin(lbl, new Insets(4, 0, 0, 0));
                lineSection.getChildren().add(lbl);
                sections.add(lineSection);
            }

        } else if (c != null && c.error != null) {
            Label warn = new Label("Could not classify this cassette.");
            warn.getStyleClass().add("hint-text");
            warn.setWrapText(true);
            box.getChildren().add(warn);
        } else {
            Label hint = new Label("Press Read Results to classify this cassette.");
            hint.getStyleClass().add("hint-text");
            hint.setWrapText(true);
            box.getChildren().add(hint);
        }

        // Metrics section
        VBox metricsSection = new VBox(6);
        javafx.scene.control.Separator sep = new javafx.scene.control.Separator();
        VBox.setMargin(sep, new Insets(6, 0, 4, 0));
        metricsSection.getChildren().add(sep);
        metricsSection.getChildren().add(compactMetric("Orientation",
                String.format(Locale.ROOT, "%.1f°", d.rotation_deg)));
        metricsSection.getChildren().add(compactMetric("Region size",
                String.format(Locale.ROOT, "%.0f × %.0f px", d.bbox_width, d.bbox_height)));
        sections.add(metricsSection);

        // Staggered reveal
        final int BASE_DELAY = 280;
        for (int i = 0; i < sections.size(); i++) {
            Node section = sections.get(i);
            section.setOpacity(0);
            box.getChildren().add(section);
            final int delay = BASE_DELAY + i * 65;
            PauseTransition pause = new PauseTransition(Duration.millis(delay));
            pause.setOnFinished(e -> {
                new Timeline(new KeyFrame(Duration.millis(200),
                    new KeyValue(section.opacityProperty(), 1.0, EASE_OUT),
                    new KeyValue(section.translateYProperty(), 0, EASE_OUT)
                )).play();
            });
            pause.play();
        }

        return box;
    }

    private Label buildAnimatedConfidenceLabel(double confidence) {
        int target = (int) Math.round(confidence * 100);
        Label label = new Label("Result confidence   0%");
        label.getStyleClass().add("metric-strong");
        final long[] startMs = {-1};
        Timeline counter = new Timeline(new KeyFrame(Duration.millis(16), e -> {
            if (startMs[0] < 0) startMs[0] = System.currentTimeMillis();
            long elapsed = System.currentTimeMillis() - startMs[0];
            double progress = Math.min(1.0, elapsed / 700.0);
            double eased = 1.0 - Math.pow(1.0 - progress, 3.0);
            int current = (int) Math.round(eased * target);
            label.setText(String.format(Locale.ROOT, "Result confidence   %d%%", current));
        }));
        counter.setCycleCount((int)(700.0 / 16) + 5);
        counter.setOnFinished(ev -> label.setText(
                String.format(Locale.ROOT, "Result confidence   %.1f%%", confidence * 100)));
        counter.play();
        return label;
    }

    private HBox probBar(String cls, double p) {
        Label name = new Label(cls);
        name.getStyleClass().add("prob-name");
        name.setMinWidth(60);

        double targetWidth = Math.max(2, 150 * Math.max(0, Math.min(1, p)));
        Region fill = new Region();
        fill.getStyleClass().add("prob-fill");
        fill.setStyle("-fx-background-color: " + verdictColor(cls) + ";");
        fill.setPrefWidth(0);
        fill.setMinHeight(7);
        fill.setPrefHeight(7);
        StackPane track = new StackPane(fill);
        track.getStyleClass().add("prob-track");
        StackPane.setAlignment(fill, Pos.CENTER_LEFT);
        HBox.setHgrow(track, Priority.ALWAYS);

        Label pct = new Label(String.format(Locale.ROOT, "%.0f%%", p * 100));
        pct.getStyleClass().add("prob-pct");
        pct.setMinWidth(36);

        HBox row = new HBox(8, name, track, pct);
        row.setAlignment(Pos.CENTER_LEFT);

        new Timeline(
            new KeyFrame(Duration.ZERO,         new KeyValue(fill.prefWidthProperty(), 0.0,         EASE_OUT)),
            new KeyFrame(Duration.millis(380),   new KeyValue(fill.prefWidthProperty(), targetWidth, EASE_OUT))
        ).play();

        return row;
    }

    private Label badge(String label) {
        Label b = new Label(label.toUpperCase(Locale.ROOT));
        b.getStyleClass().addAll("badge", badgeClass(label));
        b.setOpacity(0);
        b.setScaleX(0.92);
        b.setScaleY(0.92);
        new Timeline(new KeyFrame(Duration.millis(220),
            new KeyValue(b.opacityProperty(), 1.0, EASE_OUT),
            new KeyValue(b.scaleXProperty(),  1.0, EASE_OUT),
            new KeyValue(b.scaleYProperty(),  1.0, EASE_OUT)
        )).play();
        return b;
    }

    private HBox compactMetric(String label, String value) {
        Label l = new Label(label);
        l.getStyleClass().add("metric-label");
        Region grow = new Region();
        HBox.setHgrow(grow, Priority.ALWAYS);
        Label v = new Label(value);
        v.getStyleClass().add("metric-value");
        HBox row = new HBox(6, l, grow, v);
        row.setAlignment(Pos.CENTER_LEFT);
        return row;
    }

    private void selectDetection(int idx) {
        if (lastResult == null || lastResult.detections == null) return;
        if (idx < 0 || idx >= lastResult.detections.size()) return;
        selectedDetection = idx;
        DetectionResult.Detection d = lastResult.detections.get(idx);
        if (d.crop_image != null) {
            Path crop = Paths.get(d.crop_image);
            if (Files.exists(crop)) setViewerImage(crop);
        } else if (lastResult.annotated_image != null) {
            Path ann = Paths.get(lastResult.annotated_image);
            if (Files.exists(ann)) setViewerImage(ann);
        }
        renderResult();
    }

    // ==================== Color helpers ====================

    private String verdictColor(String label) {
        if (label == null) return "#AEAEB2";
        switch (label.toLowerCase(Locale.ROOT)) {
            case "positive": return POSITIVE_COLOR;
            case "negative": return NEGATIVE_COLOR;
            case "invalid":  return INVALID_COLOR;
            default:         return "#AEAEB2";
        }
    }

    private String verdictGlow(String label) {
        if (label == null) return "rgba(0,196,154,0.14)";
        switch (label.toLowerCase(Locale.ROOT)) {
            case "positive": return "rgba(255,59,48,0.18)";
            case "negative": return "rgba(48,209,88,0.18)";
            case "invalid":  return "rgba(255,149,0,0.18)";
            default:         return "rgba(174,174,178,0.12)";
        }
    }

    /** Stronger glow used for the image-stack highlight (no border). */
    private String verdictGlowStrong(String label) {
        if (label == null) return "rgba(0,196,154,0.35)";
        switch (label.toLowerCase(Locale.ROOT)) {
            case "positive": return "rgba(255,59,48,0.62)";
            case "negative": return "rgba(48,209,88,0.58)";
            case "invalid":  return "rgba(255,149,0,0.58)";
            default:         return "rgba(174,174,178,0.32)";
        }
    }

    private String verdictBg(String label) {
        if (label == null) return "#FFFFFF";
        switch (label.toLowerCase(Locale.ROOT)) {
            case "positive": return "#FFF2F1";
            case "negative": return "#F1FBF4";
            case "invalid":  return "#FFF8EE";
            default:         return "#FFFFFF";
        }
    }

    /** Image viewer is always clean — all result colors stay inside the Analysis panel. */
    private void updateImageBorder(String label) {
        if (imageStack == null) return;
        // Never apply color to the viewer container — it stays white/glass at all times
        imageStack.setStyle(null);
        if (noRoiOverlay != null) noRoiOverlay.setVisible("noroi".equals(label));
    }

    private String badgeClass(String label) {
        if (label == null) return "badge-unknown";
        switch (label.toLowerCase(Locale.ROOT)) {
            case "positive": return "badge-positive";
            case "negative": return "badge-negative";
            case "invalid":  return "badge-invalid";
            default:         return "badge-unknown";
        }
    }

    // ==================== Processing state ====================

    private void setProcessing(boolean loading) {
        if (loading) {
            detectButton.setDisable(true);
            readResultButton.setDisable(true);
            // Highlight active button
            if (activeActionButton == detectButton) {
                detectButton.getStyleClass().add("seg-active-blue");
            } else if (activeActionButton == readResultButton) {
                readResultButton.getStyleClass().add("seg-active-green");
            }
            renderShimmerSkeleton();
            if (pulseTimeline != null) pulseTimeline.stop();
            pulseTimeline = new Timeline(
                new KeyFrame(Duration.ZERO,         new KeyValue(statusLabel.opacityProperty(), 1.0, EASE_OUT)),
                new KeyFrame(Duration.millis(900),  new KeyValue(statusLabel.opacityProperty(), 0.35, EASE_OUT)),
                new KeyFrame(Duration.millis(1800), new KeyValue(statusLabel.opacityProperty(), 1.0, EASE_OUT))
            );
            pulseTimeline.setCycleCount(Animation.INDEFINITE);
            pulseTimeline.play();
        } else {
            // Remove active highlight
            detectButton.getStyleClass().remove("seg-active-blue");
            readResultButton.getStyleClass().remove("seg-active-green");
            activeActionButton = null;
            stopShimmerAnimation();
            if (pulseTimeline != null) { pulseTimeline.stop(); pulseTimeline = null; }
            animateStatusLabel();
            statusLabel.setOpacity(1.0);
            updateUiState();
        }
    }

    private void renderShimmerSkeleton() {
        if (detailsContainer == null) return;
        if (shimmerTimeline != null) shimmerTimeline.stop();

        VBox card1 = new VBox(10,
            shimmerLine(100, 12), shimmerLine(160, 9), shimmerLine(130, 9));
        card1.getStyleClass().add("shimmer-card");

        VBox card2 = new VBox(10,
            shimmerLine(80, 12), shimmerBlock(68), shimmerLine(180, 8), shimmerLine(140, 8));
        card2.getStyleClass().add("shimmer-card");

        detailsContainer.getChildren().setAll(card1, card2);
        animateIn(card1);
        PauseTransition p = new PauseTransition(Duration.millis(65));
        p.setOnFinished(e -> animateIn(card2));
        p.play();

        shimmerTimeline = new Timeline(
            new KeyFrame(Duration.ZERO,
                new KeyValue(card1.opacityProperty(), 0.55, EASE_OUT),
                new KeyValue(card2.opacityProperty(), 0.32, EASE_OUT)),
            new KeyFrame(Duration.millis(950),
                new KeyValue(card1.opacityProperty(), 1.0, EASE_OUT),
                new KeyValue(card2.opacityProperty(), 0.75, EASE_OUT)),
            new KeyFrame(Duration.millis(1900),
                new KeyValue(card1.opacityProperty(), 0.55, EASE_OUT),
                new KeyValue(card2.opacityProperty(), 0.32, EASE_OUT))
        );
        shimmerTimeline.setCycleCount(Animation.INDEFINITE);
        shimmerTimeline.play();
    }

    private void stopShimmerAnimation() {
        if (shimmerTimeline != null) { shimmerTimeline.stop(); shimmerTimeline = null; }
    }

    private Region shimmerLine(double width, double height) {
        Region r = new Region();
        r.getStyleClass().add("shimmer-line");
        r.setPrefWidth(width); r.setMaxWidth(width);
        r.setPrefHeight(height); r.setMinHeight(height);
        return r;
    }

    private Region shimmerBlock(double height) {
        Region r = new Region();
        r.getStyleClass().add("shimmer-block");
        r.setPrefHeight(height); r.setMinHeight(height);
        r.setMaxWidth(Double.MAX_VALUE);
        return r;
    }

    private void animateStatusLabel() {
        if (statusLabel == null) return;
        statusLabel.setTranslateY(6);
        statusLabel.setOpacity(0);
        new Timeline(new KeyFrame(Duration.millis(180),
            new KeyValue(statusLabel.translateYProperty(), 0,   EASE_OUT),
            new KeyValue(statusLabel.opacityProperty(),   1.0, EASE_OUT)
        )).play();
    }

    private void animateToolbarEntrance() {
        if (overlayLeftToolbar == null) return;
        for (HBox bar : new HBox[]{overlayLeftToolbar, overlayRightToolbar}) {
            if (bar == null) continue;
            bar.setOpacity(0);
            bar.setTranslateY(-12);
        }
        PauseTransition delay = new PauseTransition(Duration.millis(90));
        delay.setOnFinished(e -> {
            for (HBox bar : new HBox[]{overlayLeftToolbar, overlayRightToolbar}) {
                if (bar == null) continue;
                new Timeline(new KeyFrame(Duration.millis(240),
                    new KeyValue(bar.opacityProperty(),    1.0, EASE_OUT),
                    new KeyValue(bar.translateYProperty(), 0,   EASE_OUT)
                )).play();
            }
        });
        delay.play();
    }

    // ==================== Button micro-animations ====================

    private void addButtonAnimations(Button btn) {
        btn.setOnMouseEntered(e -> new Timeline(new KeyFrame(Duration.millis(120),
            new KeyValue(btn.translateYProperty(), -1.5, EASE_OUT)
        )).play());
        btn.setOnMouseExited(e -> new Timeline(new KeyFrame(Duration.millis(160),
            new KeyValue(btn.translateYProperty(), 0, EASE_OUT)
        )).play());
        btn.setOnMousePressed(e -> new Timeline(new KeyFrame(Duration.millis(80),
            new KeyValue(btn.scaleXProperty(), 0.97, EASE_OUT),
            new KeyValue(btn.scaleYProperty(), 0.97, EASE_OUT)
        )).play());
        btn.setOnMouseReleased(e -> new Timeline(new KeyFrame(Duration.millis(180),
            new KeyValue(btn.scaleXProperty(),     1.0, EASE_OUT),
            new KeyValue(btn.scaleYProperty(),     1.0, EASE_OUT),
            new KeyValue(btn.translateYProperty(), 0,   EASE_OUT)
        )).play());
    }

    // ==================== Misc ====================

    private void showAlert(String msg) {
        Platform.runLater(() -> {
            Alert a = new Alert(Alert.AlertType.WARNING, msg);
            a.setHeaderText(null);
            a.showAndWait();
        });
    }

    private static String resolvePythonExecutable() {
        String env = System.getenv("LFT_PYTHON");
        if (env != null && !env.isBlank()) return env.trim();
        return "python";
    }

    public static void main(String[] args) { launch(args); }
}
