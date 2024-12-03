# AI_Dashboard

### A dashboard for Matter Tunnel that can be queried via natural language

## Our Team

- Dongwook Kim: Dept.of Information Systems dongwook1214@gmail.com
- Giram Park: Dept.of Information Systems kirammida@hanyang.ac.kr
- Seoyoon Jung: Dept.of Information Systems yoooonnn@naver.com

- Jisu Sin (Not a student in AI class): Dept.of Information Systems sjsz0811@hanyang.ac.kr

## Proposal

Our team introduces "AI_dashboard," which revolutionizes how organizations interact with and derive insights from blockchain data. This solution combines the power of artificial intelligence with blockchain technology to facilitate data access and analysis while maintaining high levels of data integrity.

The core of AI_dashboard is a locally hosted desktop application that allows users to query blockchain data using natural language. Leveraging fine-tuned AI models, our system transforms intuitive natural language inputs into precise data query commands, eliminating traditional barriers between users and data. This approach makes blockchain data analysis accessible to team members of all technical levels, from skilled developers to business analysts.

The platform's architecture is built on direct blockchain integration that connects to Hyperledger Fabric through gRPC, ensuring data integrity by eliminating potential points of corruption in the data transmission process. The fine-tuned AI model translates natural language requests into executable queries, allowing users to extract meaningful insights from blockchain data without requiring expertise in programming languages or blockchain technology.

This document will detail the AI model development process for AI_dashboard. We will explain step-by-step the entire process, from data augmentation using advanced language models like GPT-4o and Claude to generate variations of human-annotated blockchain query datasets - allowing us to systematically expand our limited initial dataset while preserving the semantic integrity of expert-validated queries - through to fine-tuning language models using this enriched data. This approach combines the efficiency of AI-powered augmentation with the reliability of human-validated blockchain queries to create a comprehensive training dataset.

Our solution eliminates intermediary-related data corruption risks, improves accessibility for non-technical users, maintains data integrity through blockchain immutability, provides real-time data analysis capabilities, and ensures secure local processing of sensitive information.

## Datasets
### 1. `simplified_generated_dataset.json`
A synthetic query-code parallel dataset containing 5000 pairs of natural language transaction queries and their corresponding Python filter code. Each entry consists of:
- Query: Natural langauge query for filtering transactions
- Code: Corresponding Python code using `TransactionFilter` class

Sample format:
```json
{
    "dataset": [
        {
            "input": "List 2 transaction to {pk} from {src_pk} setup function after {timestamp} recent",
            "output": "txn.by_pk('{pk}').by_src_pk('{src_pk}').by_func_name('setup').by_timestamp('{timestamp}').by_order(1).get_result(2)"
        },
    ]
}
```

### 2. `simplified_augmented_dataset.json`
An augmented version of the query-code dataset using Natural Language Processing (NLP) techniques to increase lingustic diversity while preserving the semantic structure. The augmentation process maintains the original code outputs while varying the natural language queries.

Sample augmented format: 
```json
{
    "dataset": [
        {
            "input": "Catch two from 302b0eec8ff93c491a933ce925dc302a654a0b0c12b6cc7719e46a7734662bb9c9edc3430091217f52d6ee8f7b10ccc0007f0744372e6181af87e6e31eb6f19687 after 1651024241",
            "output": "print(TransactionFilter(data).by_src_pk('{src_pk}}').by_timestamp('{timestamp}}').get_result()[:2])"
        },
    ]
}
```

## Methodology
### Dataset Generation
The synthetic query-code dataset is generated using `TransactionFilterDatasetGenerator` class with the following parameters and logic.
#### 1. Query Generation Components:
- Commands: `Fetch`, `Get`, `Query`, `Load`, `Read`, `Pull`, `Show`, `List`
- Function types: `setup`, `on`, `off`
- Sort orders: `recent`, `earliest`
- Count: number of transactions
- Random public key: 130-character hexadecimal strings
- Timestamps: Unix timestamps between 1600000000 and 1700000000
#### 2. Query Generation Rules:
- Each query randomly includes combinations of:
    - Target public key (to [pk])
    - Source public key (from [src_pk])
    - Function type (setup/on/off)
    - Timestamp filter (before/after [timestamp])
    - Count (1-10)
    - Sort order (recent/earliest)
- At least one filter condition is guranteed for each query

#### 3. Code Generation Logic
- Translates natural language queries into `TransactionFilter` method chains
- Applies filters in the following order:
    - `.by_pk()` for target public key
    - `.by_src_pk()` for source public key
    - `.by_func_name()` for function type
    - `.by_timestamp()` for timestamp filter
    - `.by_order()` for ordering
    - Transaction count
The dataset generation ensures diverse combination of filtering conditions while maintaining consistent translation patterns between natural language queries and their corresponding Python filter code.

### Data Augmentation
The dataset is augmented using `QueryAugmenterNlpAug` class to increase lingustic diversity while preserving the semantic structure. The augmentation process includes:
#### 1. Initialization and Preprocessing
- Downloads required NLTK packages (wordnet, averaged_perceptron_tagger, averaged_perceptron_tagger_eng, punkt)
- Initializes regex patterns for identifying:
    - 130-character hexadecimal values
    - from/to patterns with hex public key
    - Unix timestamps
    - Function types (setup/on/off function)
- Defines preserved keywords and number variations

#### 2. Preservation Rules:
Critical components are preserved:
- Hexadecimal public key
- Function type (setup/on/off function)
- Timestamp (10-digit numbers)
- Directional keywords (to/from)
- All preserved keywords are added to stopwords list to prevent modification

#### 3. Augmentation Techniques:
This system employs three different augmenters with the following configurations:
##### WordNet Synonym Substitution
- Using WordNet to replace query words and general terms with synonyms
- Configurations:
    - Probability of augmentation: 0.2 (20% chance of word replacement)
    - Minimum words to augment: 1
    - Preserves critical keywords via stopwords
- Examples:
    - Original: "Show two transactions from [hex] after timestamp"
    - Augmented: "Display two transactions from [hex] after timestamp"

##### BERT Contextual Word Substitution (BERT)
- Utilizes bert-base-uncased model for context-aware word replacements
- Configuration:
    - Model: bert-base-uncased
    - Probability of substitution: 0.2
    - Minimum words to substitute: 1
    - Action: "substitute"
    - Preserves stopwords
- Examples:
    - Original: "Get transactions to [hex] setup function"
    - Augmented: "Fetch records to [hex] setup function"

##### Random Word Swap
- Performs controlled random swapping of words to create syntactic variations
- Configuration:
    - Probability of swap: 0.3
    - Minimum words to swap: 1
    - Action: "swap"
    - Preserves stopwords and critical structure
- Examples:
    - Original: "Query recent transactions from [hex]"
    - Augmented: "Recent transactions query from [hex]"

Each augmentation technique is applied sequentially, with careful preservation of critical elements like:
- Hexadecimal public keys
- Function types (setup/on/off function)
- Timestamps
- Directional indicators (to/from)

![Data Augmentation](https://github.com/Winter-Zzzz/AI_dashboard/blob/main/image/data_augmentation_result.png?raw=true)

#### 4. Quality Control

The system ensures data quality through three main steps:

##### Input Validation
- Only one function type (setup/on/off) allowed per query
- Function keywords must follow pattern: `(off|on|setup) function`
- Examples:
  - Valid: "Get setup function data"
  - Invalid: "Get setup on function data"

##### Text Cleaning
1. Output Text Processing:
    - Removes spaces in method chains
    - Example:
        - Before: "txn.by_func_name('setup') . get_result()
        - After: "txn.by_func_name('setup').get_result()
    - Cleans parentheses spacing
    - Strips whitespace
2. Input Text Cleaning:
   - Removes excessive whitespace
   - Preserves critical keywords (function names, hex values, timestamps)

##### Data Validation
- Filters out invalid augmentations (containing 'UNK' tokens)
- Removes duplicate input-output pairs
- Processes in batches of 512 for efficiency
- Includes error handling and logging

This multi-layered quality control process ensures that the augmented dataset maintains high standards of quality and usefulness for training purposes while preserving the essential characteristics of the original queries.

### Model Training and Fine-tuning
Our model architecture is based on the T5 (Text-to-Text Transfer Transformer), specifically configured for the task of translating natural language queries into transaction filter code. The training process incorporates the following specifications:

#### 1. Model Configuration
- Base Model: T5-small
- Maximum Sequence Length: 256 tokens
- Maximum Generation Length: 256 tokens
- Training Device: GPU with CUDA support (CPU fallback available)

#### 2. Token Processing
- Special Tokens
    ```python
    [
        '<hex>',
        '</hex>',
        '<time>',
        '</time>'
    ]
    ```
- Token Embedding Resizing: Accomodates additional special tokens
- Prefix Space Addition: Right-side padding with EOS token
- Attention Masking: Handles variable length sequences

#### 3. Training Parameters
- Learning Rate: 5e-5
- Batch Size: 8 (with gradient accumulation steps of 4)
- Weight Decay: 0.01
- Number of Epochs: 30
- Early Stopping Patience: 7
- Gradient Clipping: 1.0
- Gradient Accumulation Steps: 4

#### 4. Generation Settings
- Beam Search Size: 5
- Length Penalty: 1.0
- No Repeat N-gram Size: 0
- Early Stopping: Enabled

#### 5. Training Process
- Data Split: 90% training, 10% validation
- Optimizer: AdamW with linear warmup scheduler
- Mixed Precision Training: Automatic mixed precision with GradScaler
- Warmup: 10% of total steps
- Checkpointing: Saves best model based on validation loss
- Progress Tracking: Real-time loss monitoring and visualization

#### 6. Implementation Details
1. **Training Tracker System**

    ```python
    class TrainingTracker:
        """
        Tracks training progress, saves status, and visualizes performance metrics
        """
    ```
    Components:
    - Real-time loss monitoring
    - Best model checkpointing
    - Progress visualization
    - Training status persistence
2. **Training Optimization**: Resource Management
    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    ```
    Optimization techniques:
        - GPU acceleration when available
        - Automoatic memory clearing before training
        - Mixed precision training with GradScaler



        - Gradient accumulation (steps=4)

3. **Training Process Control**
: Early Stopping Implentation
    ```python
    if no_improve >= patience:  # patience = 7
    print("🛑 Early stopping triggered!")
    break
    ```
- Patience-based monitoring (7 epochs)
- Best model preservation
- Training efficiency optimization

4. **Performance Monitoring**: Metrics Tracking
- Training Loss
    - Batch-level monitoring
    - Running average calculation
    - Gradient accumulation adjustments
- Validationn Process
    - Per-epoch validation
    - Best model selection
    - Early stopping triggers

5. **Stauts Management**
    ```json
    {
        "current_epoch": current,
        "best_loss": loss_value,
        "last_train_loss": train_loss,
        "total_improvements": count
    }
    ```
    ![training_status](https://github.com/Winter-Zzzz/AI_dashboard/blob/main/image/training_status.png?raw=true)

6. **Validation Sampling**

Each epoch includes validation output sampling for quality control
- Input query verification
- Output code structure validation

The training configuration emphasizes efficiency and performance through the use of gradient accumulation, mixed precision training, and careful hyperparameter selection. The relatively small sequence length of 256 tokens is sufficient for our query-to-code translation task while enabling faster training and inference.

### Installation and Setup
Follow these steps to set up and run the project:
1. Download Required Model File: [models.safetensors](https://drive.google.com/file/d/186jAWNLmJbB1TuqCvWWg1eY4D7Q3aDW6/view)

2. Place it in the following directory: `ai/models/best_models`

3. Grant execution permissions to the setup script
    ```bash
    chmod +x setup.sh
    ```

4. Run the setup script
    ```bash
    ./setup.sh
    ```

### API Server Implementation
#### FastAPI Server Configuration
- Implements a FastAPI server with CORS middleware enabled
- Server runs on host "0.0.0.0" and port 8000
- Allows cross-origin requests from all origins with full method and header access

#### API Endpoints
`/api/query-transactions` (GET)

Response Format:
```json
{
    "status": "success",
    "data": {
        "transactions": [...],
        "generated_code": "transformed_code_string",
        "query_text": "original_query_string"
    }
}
```

## Evaluation
After training for 30 epochs, our model demonstrated robust performance and consistent learning characteristics. Here's a comprehensive analysis of the results.

1. **Training Performances**
    - Best Loss: 0.00369
    - Average Training Loss: 0.0054
    - Average Validation Loss: 0.0043
    - Total Training Epochs: 30
   - Total Improvements: 30

2. **Convergence Analysis**

    ![training_progress](https://github.com/Winter-Zzzz/AI_dashboard/blob/main/image/training_progress.png?raw=true)

    The system generates real-time training progress plots showing:
    - Epoch-wise training loss
    - Improvement points highlighted
    - Learning rate scheduling effects
    - Training Progress Plot:
        - X-axis: Epochs
        - Y-axis: Loss values
        - Green dots: Points of improvement
        - Continuous line: Training loss trend

    The training progress plot demonstrates strong convergence characteristics:
    - Initial rapid learning phase with loss dropping from 1.2 to 0.4 in first 2 epochs
    - Steady improvement phase between epochs 2-10
    - Fine-tuning phase after epoch 10 with consistent minor improvements
    - No signs of overfitting (validation loss consistently below training loss)
    - Smooth convergence curve without significant fluctuations
    
    ![training_progress](https://github.com/Winter-Zzzz/AI_dashboard/blob/main/image/training_progress.png?raw=true)

3. **Model Validation**

    Each epoch includes sample validation outputs:

    ![Validation](https://github.com/Winter-Zzzz/AI_dashboard/blob/main/image/validation.png?raw=true)

    The model demonstrates:
    - Correct preservation of query structure
    - Accurate handling of special tokens and timestamps
    - Proper translation of natural language to filter chain syntax
    - Appropriate parameter formatting and method ordering

4. **Performance Summary**
    The evaludation results indicate that
    - The model successfully learned the natural language to code translation task
    - Training was stable and efficient without triggering early stopping
    - The low validation loss (0.0043) suggests good generalization capability
    - The model maintains consistent performance across different query patterns and complexities

These results validate that our model is well-suited for the AI_dashboard system's requirements of translating natural language queries into blockchain transaction filter code.

## Conclusion

This paper presented AI_dashboard, an innovative solution that bridges the gap between natural language interaction and blockchain data analysis. Our implementation successfully demonstrates several key technical achievements and system benefits while opening avenues for future development.

### Technical Achievements

1. **Advanced Data Generation and Augmentation**
   - Successfully generated a comprehensive synthetic dataset of 5000 query-code pairs
   - Implemented sophisticated augmentation techniques using WordNet, BERT, and random word swap methods
   - Achieved high-quality data preservation through multi-layered quality control processes

2. **Effective Model Architecture and Training**
   - Successfully fine-tuned T5-small model for query-to-code translation
   - Achieved impressive convergence with best loss of 0.00369
   - Implemented efficient training optimizations including gradient accumulation and mixed precision training
   - Demonstrated stable learning characteristics across 30 epochs without overfitting

3. **Direct Blockchain Integration**
   - Developed a scalable FastAPI server architecture with gRPC connection
   - Implemented comprehensive error handling and validation
   - Ensures data integrity through direct blockchain access without intermediaries

### System Benefits

1. **Enhanced Accessibility**
   - Democratized blockchain data access for non-technical users
   - Eliminated the need for specialized query language knowledge
   - Reduced the learning curve for new team members

2. **Guaranteed Data Integrity**
   - Direct blockchain integration through gRPC ensures data authenticity
   - No intermediary systems that could potentially compromise data
   - Leverages blockchain's inherent immutability for reliable data access

3. **Operational Efficiency**
   - Real-time natural language query processing
   - Direct blockchain data access for immediate insights
   - Streamlined workflow for data analysis tasks

### Current Limitations and Future Work

1. **Computational Resources and Data Scale**
   - Current implementation was constrained by Google Colab's GPU usage limits
   - Training data size was optimized for available computational resources
   - Access to more powerful GPU infrastructure would enable:
     - Larger-scale data augmentation
     - More extensive model training
     - Experimentation with larger language models
     - Potential for significantly improved performance metrics

2. **Model Enhancement**
   - Expand the training dataset with more complex query patterns
   - Explore larger language models for improved accuracy
   - Implement continuous learning capabilities with user feedback

3. **Feature Extensions**
   - Develop support for more complex analytical queries
   - Add visualization capabilities for query results
   - Implement batch processing for large-scale analyses

4. **Performance Optimization**
   - Enhance query processing efficiency
   - Optimize for large-scale deployments
   - Expand blockchain integration capabilities

This project successfully demonstrates the potential of combining AI with blockchain technology to create more accessible and efficient data analysis tools. While our current implementation was constrained by computational resources in the Google Colab environment, the achieved results show significant promise. With access to more powerful GPU infrastructure, the system could be further enhanced through larger-scale data augmentation and more extensive model training. The strong foundation established in this work, particularly the direct blockchain integration ensuring data integrity, provides an excellent starting point for both academic research and practical applications in the blockchain industry.