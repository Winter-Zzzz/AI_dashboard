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

#### 4. Quality Control
The system implements a comprehensive quality control process to ensure the integrity and usefulness of augmented data:

##### Function Type Validation
- Enforces strict rules for function-related content:
  - Only one function type allowed per query (setup/on/off)
  - Validates correct format using regex pattern: `\b(off|on|setup)\s+function\b`
  - Prevents duplicate function type words in a single query
  - Example of valid: "Get setup function data"
  - Example of invalid: "Get setup on function data"

##### Text Cleaning and Normalization
1. Basic Text Cleaning:
   - Removes excessive whitespace
   - Standardizes spacing around special characters
   - Trims leading and trailing spaces

2. Function Type Standardization:
   - Converts function references to standardized format
   - Transforms: "setup function" → "by_func_name('setup')"
   - Handles variations: "off function", "on function"

3. TransactionFilter Syntax Optimization:
   - Removes unnecessary spaces around method chains
   - Standardizes spacing around parentheses
   - Example: "TransactionFilter(data) . by_pk ( )" → "TransactionFilter(data).by_pk()"

##### Filtering Process
The augmentation results undergo multiple validation steps:

1. Pre-augmentation Validation:
   - Verifies input/output data types are correct (lists of strings)
   - Checks for presence of required fields
   - Validates data structure integrity

2. During Augmentation Filtering:
   - Removes results containing 'UNK' tokens
   - Ensures function type validity
   - Maintains preserved keyword integrity

3. Post-augmentation Cleanup:
   - Eliminates duplicate input-output pairs
   - Verifies format consistency
   - Ensures all preserved components remain intact

##### Data Consistency Checks
1. Structure Validation:
   - Verifies JSON format compliance
   - Checks for required fields in each entry
   - Validates data type consistency

2. Content Validation:
   - Ensures public keys maintain 130-character length
   - Verifies timestamp format (10-digit numbers)
   - Checks preservation of critical keywords

##### Output Generation
1. Final Validation:
   - Performs final verification of augmented pairs
   - Ensures proper cleaning and formatting
   - Validates semantic consistency

2. Statistics Generation:
   - Records number of successfully augmented pairs
   - Tracks augmentation success rate
   - Documents filtering statistics

3. Storage:
   - Saves validated data in structured JSON format
   - Maintains clear input-output mapping
   - Includes metadata about augmentation process

This multi-layered quality control process ensures that the augmented dataset maintains high standards of quality and usefulness for training purposes while preserving the essential characteristics of the original queries.

#### Model Training and Fine-tuning
The model architecture is based on Text-to-Text Transfer Transformer (T5), fine-tuned for the query-to-code generation task with the following specifications:

1. Model Configuration

    - Base Model: T5-base
    - Max Sequence Length: 128 tokens
    - Training Device: GPU (CUDA) if available, CPU otherwise
    - Learning Rate: 2e-4
    - Batch Size: 4
    - Weight Decay: 0.05
    - Number of Epochs: 20
    - Special Tokens: Added domain-specific tokens for code generation

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
            ')',
            '<HEX_ADDRESS>',
            '<TIMESTAMP>'
        ]
        ```
2. Generation Parameters:
    - Number of Beams: 10
    - Temperature: 0.7
    - Top P: 0.9
    - Maximum Generation Length: 128 tokens

3. Training Process:
    - Early Stopping: Implemented with patience=5
    - Optimizer: AdamW
    - Loss Function: Cross-entropy loss
    - Model Checkpointing: Saves best model based on validation loss
    - Input Format: Natural language queries with special tokens
    - Output Format: Python code using `TransactionFilter` class with special tokens
    - Gradient Clipping: 0.5
    - Warm Up Ratio: 0.1

4. Token Processing:
    - Token Embedding Resizing: Accomodates additional special tokens
    - Gradient Updates: Per batch with zero_grad()
    - Attention Mask: Handles variable length sequences
    - Prefix Space Addition: Enhanced tokenization for special tokens 

## Evaluation

tbd ...

## Conclusion

tbd ...
