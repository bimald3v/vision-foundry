# Warehouse Object Detection Workflow

## System Architecture

```mermaid
graph TD
    subgraph Input
        A[Video Feed] --> B[Frame Extraction]
        B --> C[Frame Preprocessing]
    end

    subgraph YOLOv8 Detection
        C --> D[Feature Extraction<br/>CSPDarknet]
        D --> E[Feature Aggregation<br/>PANet]
        E --> F[Object Detection<br/>Decoupled Heads]
    end

    subgraph Post Processing
        F --> G[Confidence Filtering<br/>threshold: 0.3]
        G --> H[Class Mapping<br/>COCO → Warehouse]
        H --> I[Bounding Box Drawing]
        I --> J[Label Annotation]
    end

    subgraph Output
        J --> K[Processed Frame]
        K --> L[Video Writer]
        L --> M[Output Video]
    end
```

## Detection Pipeline Flow

```mermaid
sequenceDiagram
    participant Video as Video Feed
    participant Detector as YOLOv8 Detector
    participant Mapper as Class Mapper
    participant Visualizer as Visualizer
    participant Output as Output Video

    Video->>Detector: Read Frame
    Note over Detector: Process frame in<br/>single forward pass
    Detector->>Mapper: Detection Results
    Note over Mapper: Map COCO classes to<br/>warehouse labels
    Mapper->>Visualizer: Mapped Detections
    Note over Visualizer: Draw bounding boxes<br/>Add labels & confidence
    Visualizer->>Output: Write Frame
    
    loop Until End of Video
        Video->>Detector: Next Frame
    end
```

## Class Mapping System

```mermaid
graph LR
    subgraph COCO Classes
        A[person]
        B[truck/car]
        C[backpack/handbag]
        D[suitcase/box]
        E[tv]
    end

    subgraph Warehouse Labels
        F[warehouse worker]
        G[forklift]
        H[warehouse equipment]
        I[cardboard box]
        J[pallet rack]
    end

    A --> F
    B --> G
    C --> H
    D --> I
    E --> J

    style F fill:#87CEEB
    style G fill:#FFD700
    style H fill:#90EE90
    style I fill:#DEB887
    style J fill:#FF6B6B
```

## Color Coding System

```mermaid
graph TD
    subgraph Object Categories
        A[Warehouse Workers] --> B[Blue #0000FF]
        C[Forklifts] --> D[Yellow #FFD700]
        E[Cardboard Boxes] --> F[Brown #8B4513]
        G[Equipment] --> H[Green #00FF00]
        I[Pallet Racks] --> J[Red #FF0000]
        K[Unknown] --> L[Gray #808080]
    end

    style B fill:#0000FF, color:#FFFFFF
    style D fill:#FFD700
    style F fill:#8B4513, color:#FFFFFF
    style H fill:#00FF00
    style J fill:#FF0000, color:#FFFFFF
    style L fill:#808080, color:#FFFFFF
```

## Performance Metrics

```mermaid
graph LR
    subgraph Processing Steps
        A[Frame<br/>Extraction] -->|~30fps| B[YOLOv8<br/>Detection]
        B -->|0.02s/frame| C[Post<br/>Processing]
        C -->|0.01s/frame| D[Frame<br/>Writing]
    end

    subgraph Performance Factors
        E[GPU<br/>Acceleration]
        F[Batch<br/>Processing]
        G[Resolution<br/>Scaling]
    end

    E --> B
    F --> B
    G --> A
```

This workflow documentation visualizes the complete pipeline of our warehouse object detection system, from video input through detection and visualization to final output. The diagrams show the system architecture, detection flow, class mapping, color coding, and performance considerations.
