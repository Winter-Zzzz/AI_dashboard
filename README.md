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
### 1. `generated_dataset.json`
A synthetic query-code parallel dataset containing 500 pairs of natural language transaction queries and their corresponding Python filter code. Each entry consists of:
- Query: Natural langauge query for filtering transactions
- Code: Corresponding Python code using `TransactionFilter` class

Sample format:
```json
{
    "dataset": [
        {
        "input": "Get 2 from {src_pk} after {timestamp}}",
        "output": "print(TransactionFilter(data).by_src_pk('{src_pk}}').by_timestamp('{timestamp}}').get_result()[:2])"
        },
    ]
}
```

### 2. `augmented_dataset.json`
An augmented version of the query-code dataset using Natural Language Processing (NLP) techniques to increase lingustic diversity while preserving the semantic structure. The augmentation process maintains the original code outputs while varying the natural language queries.

Sample augmented queries for the same code: 
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
- Random public key: 130-character hexadecimal strings
- Timestamps: Unix timestamps between 1600000000 and 1700000000
#### 2. Query Generation Rules:
- Each query randomly includes combinations of:
    - Target public key (to [pk])
    - Source public key (from [src_pk])
    - Function type (setup/on/off)
    - Timestamp filter (before/after [timestamp])
    - Transaction Count (1-10)
    - Sort order (recent/earliest)
- At least one filter condition is guranteed for each query

#### 3. Code Generation Logic
- Translates natural language queries into `TransactionFilter` method chains
- Applies filters in the following order:
    - `.by_pk()` for target public key
    - `.by_src_pk()` for source public key
    - `.by_func_name()` for function type
    - `.by_timestamp()` for timestamp filter
    - `.sort()` for ordering
    - Transaction count
The dataset generation ensures diverse combination of filtering conditions while maintaining consistent translation patterns between natural language queries and their corresponding Python filter code.

### Data Augmentation
The dataset is further augmented using `QueryAugmenterNlpAug` class to increase lingustic diversity while preserving the semantic structure. The augmentation process includes:
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
This system employs four different augmenters with the following configurations:
##### Synonym Substitution
- Using WordNet to replace query words and general terms with synonyms
- Configurations:
    - Probability of augmentation: 0.3 (30% chance of word replacement)
    - Minimum words to augment: 1
    - Preserves critical keywords via stopwords
- Example
    - "Query" → "Retrieve"
    - "Show" → "Display"
    - "Recent" → "Latest", "Newest"

##### Contextual Word Substitution (BERT)
- Utilizes bert-base-uncased model for context-aware word replacements
- Configuration:
    - Model: bert-base-uncased
    - Probability of substitution: 0.3
    - Minimum words to substitute: 1
    - Action: "substitute"
    - Preserves stopwords
- Examples:
    - "Get transactions from" → "Retrieve records from"
    - "Show data after" → "Display entries after"

##### Contextual Word Insertion (BERT)
- Uses bert-base-uncased model to insert contextually appropriate words
Configuration:
    - Model: bert-base-uncased
    - Probability of insertion: 0.3
    - Minimum words to insert: 1
    - Action: "insert"
    - Preserves stopwords
- Examples:
    - "Get from [src_pk]" → "Get all transactions from [src_pk]"
    - "Show after timestamp" → "Show all records after timestamp"

##### Random Word Swap
- Performs controlled random swapping of compatible words to create syntactic variations
- Configuration:
    - Probability of swap: 0.3
    - Minimum words to swap: 1
    - Action: "swap"
    - Preserves stopwords and critical structure
- Examples:
    - "Query recent transactions from [src_pk]" → "Recent transactions query from [src_pk]"
    - "Get earliest data after timestamp" → "After timestamp get earliest data"

Each augmentation technique is applied sequentially, with careful preservation of critical elements like:
- Hexadecimal public keys
- Function types (setup/on/off function)
- Timestamps
- Directional indicators (to/from)

The probability settings (0.3) ensure moderate augmentation while maintaining query comprehensibility and functional equivalence. All augmented outputs are validated to ensure they maintain the correct semantic mapping to their corresponding filter code.

![Data Augmentation ](image.png)

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
   - Converts function references: "on function" → "on" in by_func_name calls
   - Removes extra spaces in method chains
   - Example: "TransactionFilter(data).by_func_name('on').get_result()"

2. Input Text Cleaning:
   - Removes excessive whitespace
   - Preserves critical keywords (function names, hex values, timestamps)

##### Data Validation
- Filters out invalid augmentations (containing 'UNK' tokens)
- Removes duplicate input-output pairs
- Verifies complete TransactionFilter syntax
- Checks hex values maintain 130-character length

This multi-layered quality control process ensures that the augmented dataset maintains high standards of quality and usefulness for training purposes while preserving the essential characteristics of the original queries.

### Model Training and Fine-tuning
Our model architecture is based on the T5 (Text-to-Text Transfer Transformer), specifically configured for the task of translating natural language queries into transaction filter code. The training process incorporates the following specifications:

1. **Model Configuration**
   - Base Model: T5-base
   - Maximum Sequence Length: 512 tokens
   - Maximum Generation Length: 512 tokens
   - Training Device: GPU with CUDA support (CPU fallback available)

2. **Token Processing**
    - Special Tokens
        ```python
        [
            'print(TransactionFilter(data)',
            'get_result()',
            '.by_pk', 
            '.by_src_pk',
            '.by_timestamp',
            '.by_func_name',
            "('setup')",
            "('on')",
            "('off')",
            '.sort(reverse=True)',
            '.sort()',
            '.',
            ')'
        ]
        ```
    - Token Embedding Resizing: Accomodates additional special tokens
    - Prefix Space Addition: Enhanced tokenization for speical tokens
    - Attention Masking: Handles variable length sequences

3. **Training Parameters**
   - Learning Rate: 1e-4
   - Batch Size: 4
   - Weight Decay: 0.01
   - Number of Epochs: 20
   - Early Stopping Patience: 5
   - Gradient Clipping: 0.5
   - Warm-up Ratio: 0.1

4. **Generation Settings**
   - Beam Search Size: 10
   - Early Stopping: Enabled

5. **Training Process**
   - Data Split: 90% training, 10% validation
   - Optimizer: AdamW with linear warmup scheduler
   - Loss Function: Cross-entropy loss
   - Gradient Updates: Per batch with zero_grad()
   - Checkpointing: Saves best model based on validation loss

## Evaluation
The evaluation framework consists of multiple components to assess model performance:

1. **Training Metrics**
    - Loss Tracking: Average training and validation loss per epoch
    - Model Convergence: Monitoring loss improvement over time
    - Example Generation: Sample outputs every 50 batches for quality assessment

2. **Performance Monitoring**
    - Training Progress Visualization
    - Real-time Loss Tracking
    - Automatic Model Checkpointing
    - Training Status Logging

3. **Quality Assessment**
    - Syntax Validation: Ensuring correct TransactionFilter code generation
    - Semantic Accuracy: Verifying query intent preservation
    -Example Generation: Regular validation of model outputs
    - Comprehensive Final Evaluation: Detailed analysis of model performance

The evaluation results are stored in both JSON format for detailed analysis and text format for human readability, allowing for thorough assessment of the model's capabilities and limitations.

## Conclusion

Our AI_Dashboard project successfully demonstrates an effective approach to bridging the gap between natural language interaction and blockchain data querying. Through careful implementation of dataset generation, augmentation, and model training, we have created a robust system that addresses key challenges in blockchain data accessibility.

### Technical Achievements

1. **Dataset Development**
   - Successfully generated a synthetic dataset of 500 query-code pairs
   - Implemented sophisticated data augmentation techniques using NLP
   - Maintained semantic integrity through careful preservation rules
   - Achieved diverse query variations while ensuring code accuracy

2. **Model Implementation**
   - Successfully fine-tuned T5-base for specialized code generation
   - Implemented custom tokenization with domain-specific tokens
   - Achieved efficient training with early stopping mechanism
   - Developed robust evaluation frameworks for performance assessment

3. **Quality Control**
   - Implemented multi-layered validation systems
   - Maintained high standards in data quality
   - Ensured semantic consistency in augmented data
   - Developed comprehensive evaluation metrics

### System Benefits

1. **Accessibility**
   - Eliminated technical barriers for non-technical users
   - Enabled natural language interaction with blockchain data
   - Simplified complex query construction
   - Reduced dependency on technical expertise

2. **Data Integrity**
   - Maintained blockchain data immutability
   - Implemented direct gRPC connection to Hyperledger Fabric
   - Eliminated intermediary-related corruption risks
   - Ensured secure local processing

### Future Work

1. **Model Enhancement**
   - Expand the training dataset with more complex query patterns
   - Implement additional augmentation techniques
   - Explore advanced model architectures
   - Optimize performance for larger-scale deployments

2. **System Development**
   - Enhance error handling mechanisms
   - Implement additional blockchain platform support
   - Develop more sophisticated monitoring tools
   - Expand query capabilities

Our implementation successfully demonstrates the viability of using natural language processing for blockchain data querying, providing a foundation for future development in making blockchain technology more accessible to non-technical users while maintaining the security and integrity of the underlying data.
