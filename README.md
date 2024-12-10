# AI_Dashboard

### A dashboard for Matter Tunnel that can be queried via natural language

## Our Team

- Dongwook Kim: Dept.of Information Systems dongwook1214@gmail.com
- Giram Park: Dept.of Information Systems kirammida@hanyang.ac.kr
- Seoyoon Jung: Dept.of Information Systems yoooonnn@hanyang.ac.kr
- Jisu Shin (SE only): Dept.of Information Systems sjsz0811@hanyang.ac.kr

## Proposal

Our team introduces "AI_dashboard," which revolutionizes how organizations interact with and derive insights from blockchain data. This solution combines the power of artificial intelligence with blockchain technology to facilitate data access and analysis while maintaining high levels of data integrity.

The core of AI_dashboard is a locally hosted desktop application that allows users to query blockchain data using natural language. Leveraging fine-tuned AI models, our system transforms intuitive natural language inputs into precise data query commands, eliminating traditional barriers between users and data. This approach makes blockchain data analysis accessible to team members of all technical levels, from skilled developers to business analysts.

The platform's architecture is built on direct blockchain integration that connects to Hyperledger Fabric through gRPC, ensuring data integrity by eliminating potential points of corruption in the data transmission process. The fine-tuned AI model translates natural language requests into executable queries, allowing users to extract meaningful insights from blockchain data without requiring expertise in programming languages or blockchain technology.

This document will detail the AI model development process for AI_dashboard. We will explain step-by-step the entire process, from data augmentation using advanced language models like GPT-4o and Claude to generate variations of human-annotated blockchain query datasets - allowing us to systematically expand our limited initial dataset while preserving the semantic integrity of expert-validated queries - through to fine-tuning language models using this enriched data. This approach combines the efficiency of AI-powered augmentation with the reliability of human-validated blockchain queries to create a comprehensive training dataset.

Our solution eliminates intermediary-related data corruption risks, improves accessibility for non-technical users, maintains data integrity through blockchain immutability, provides real-time data analysis capabilities, and ensures secure local processing of sensitive information.

## Datasets

### 1. `core_dataset.json`

A synthetic query-code parallel dataset containing 10,000 pairs of natural language transaction queries and their corresponding Python filter code. Each entry consists of:

- Query: Natural language query for filtering transactions
- Code: Corresponding Python code using `TransactionFilter` class

Sample format:
```json
{
    "dataset": [
        {
            "input": "Get three transactions from [pk] after [timestamp]",
            "output": "txn.by_pk(-1).by_src_pk('[pk]').by_func_name(-1).after('[timestamp]').by_order(0).get_result(3)"
        },
        {
            "input": "Get most recent transaction by [pk]",
            "output": "txn.by_pk(-1).by_src_pk('[pk]').by_func_name(-1).by_order(1).get_result(1)[0]"
        }
    ]
}
```

### 2. `augmentation_template_dataset.json`

An augmentation dataset of 1500 additional pairs designed to enhance the model's handling of recent order variations:
- 600 standard query-code pairs
- 900 pairs (225 each) specifically for `most recent`, `first recent`, `second recent`, and `third recent` variations

Sample format:
```json
{
    "dataset": [
        {
            "input": "Get most recent transaction to [pk]",
            "output": "txn.by_pk('[pk]').by_src_pk(-1).by_func_name(-1).by_order(1).get_result(1)[0]"
        },
        {
            "input": "Show second recent transaction by [pk]",
            "output": "txn.by_pk(-1).by_src_pk('[pk]').by_func_name(-1).by_order(1).get_result(1)[1]"
        }
    ]
}
```

### 3. `augmented_dataset.json`

An expanded dataset created by applying augmentation techniques to the core and template datasets, growing from 1,500 base examples to 3,094 augmented examples. Generated through BERT-based contextual substitution and random word swapping while preserving critical patterns like public keys, timestamps, and function names.

Sample format:
```json
{
    "dataset": [
        {
            "input": "All get latest function getCoordinate",
            "output": "txn.by_pk(-1).by_src_pk(-1).by_func_name('getCoordinate').by_order(1).get_result(-1)"
        },
        {
            "input": "Load ten oldest feedback function",
            "output": "txn.by_pk(-1).by_src_pk(-1).by_func_name('feedback').by_order(0).get_result(10)"
        }
    ]
}
```

Key characteristics:
- Maintains semantic equivalence with original queries
- Preserves critical components (public keys, timestamps, function names)
- Generated through BERT contextual substitution (30% probability)
- Includes word swap variations (30% probability)
- Filtered to remove any entries containing UNK tokens
- Maintains consistent output code for semantically equivalent queries


## Methodology

### Dataset Generation

The synthetic query-code dataset is generated using `TransactionFilterDatasetGenerator` class with the following parameters and logic.

#### 1. Query Generation Components:

- Commands: `Fetch`, `Get`, `Query`, `Load`, `Read`, `Pull`, `Show`, `List`
- Sort Orders: `latest`, `oldest`, `recent`, `earliest`, `most recent`, `last`
- Transaction Count: Numbers 1-10 (expressed as digits or words) or 'all'
- Recent Order Variations:
    - `most recent` / `first recent`: Returns the latest transaction (equivalent to [0] index)
    - `second recent`: Returns the second latest transaction (equivalent to [1] index)
    - `third recent`: Returns the third latest transaction (equivalent to [2] index)
- Filter Conditions:
    - Destination public key (to [public_key])
    - Source public key (from/by [public_key])
    - Function names (e.g., `setAnimal`, `getCoordinate`, `getTemperature`)
    - Timestamp filters (before/after/between [timestamp])

#### 2. Input Query Generation Rules:

- Each query randomly includes combinations of:
    - Command word (Required)
    - Transaction count (optional)
    - Sort order (optional)
    - Filter conditions (at least one required)
        - Public key filters (to/from/by)
        - Function name
        - Timestamp constraints

#### 3. Code Generation Rules

The generator translates natural language queries into method chains following this pattern:
```python
txn.by_pk(target_pk)
   .by_src_pk(source_pk)
   .by_func_name(function_name)
   .after(timestamp)
   .before(timestamp)
   .by_order(order)
   .get_result(count)
```

Special cases:
- Missing filters are set to `-1`
- `count` is set to `-1` for `all` or plural queries without specific count
- Order is set to `1` for latest/recent and `0` for oldest/earliest
- Recent variations append specific index:
```python
# Most recent / First recent
.get_result(1)[0]
# Second recent
.get_result(1)[1]
# Third recent
.get_result(1)[2]
```
#### 4. Usage
```python
generator = TransactionFilterDatasetGenerator()

# Generate base training dataset (10,000 pairs)
generator.generate_dataset(10000)  # Generates standard query-code pairs for training

# Generate augmentation template dataset (500 pairs)
# - 400 standard query-code pairs
# - 100 pairs for recent order variations (25 each case)
generator.generate_dataset(400)     
generator.generate_most_recent_dataset(25)  # Generates most/first/second/third recent cases

# Save dataset
with open('path/to/dataset.json', 'w') as json_file:
    json.dump(generator.dataset, json_file, indent=4)
```
Core dataset (10,000 pairs):
- The dataset ensures comprehensive coverage of query patterns and filtering combinations while maintaining consistent translation between natural language and code.
Augmentation template dataset (500 pairs):
- This template dataset focuses on specific recent-order variations and complex query patterns, designed to serve as a foundation for data augmentation techniques and fine-tuning

### Model Training 

#### 1. Model Architecture & Configuration
- Base Model: T5-small with custom configuration
- Device: CUDA GPU with CPU fallback support
- Implemented automatic device detection and memory management
    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    ```

#### 2. Training Infrastructure
**Dataset Management**
The `QueryDataset` class has been enhanced with:
- Text normalization methods
- Special token handling
- Efficient padding and truncation strategies
```python
class QueryDataset(Dataset):
    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
```

**Training Tracker Implementation**
`TrainingTracker` class for monitoring training progress:
- Loss history tracking
- Progresss visualization
- Training status persistence
- Automated checkpoint management
```python
class TrainingTracker:
    def update(self, epoch, train_loss):
        improved = train_loss < self.best_loss
        self.training_history['improvement'].append(improved)
        if improved:
            self.best_loss = train_loss
        self.plot_progress()
        self.save_status()
        return improved
```

#### 3. Training Process Optimizations
**Memory Management**
- Implemented automatic CUDA cache clearing
- Optimized batch processing with gradient accumulation
- Added mixed precision training
```python
with autocast(device_type='cuda'):
    output = model(
        input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device), 
        labels=batch['labels'].to(device)
    )
    loss = output.loss / config.ACCUMULATION_STEPS
```

**Gradient Handling**
- Implemented gradient accumulation steps
- Added gradient clipping
- Optimized scheduler updates
```python
if (i + 1) % config.ACCUMULATION_STEPS == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()
```

#### 4. Model Checkpointing
Enhanced checkpoint saving with comprehensive state preservationa;
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'tokenizer_vocab': tokenizer.get_vocab(),
    'tokenizer_special_tokens_map': tokenizer.special_tokens_map,
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'epoch': epoch,
    'loss': avg_val_loss,
}
```

#### 5. Validation Process
Improved validation with example generation:
```python
generated = model.generate(
    input_ids=batch['input_ids'][0:1].to(device),
    attention_mask=batch['attention_mask'][0:1].to(device),
    max_length=config.MAX_GEN_LENGTH,
    num_beams=config.NUM_BEAMS,
    length_penalty=config.LENGTH_PENALTY,
    no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
    early_stopping=config.EARLY_STOPPING
)
```

#### 6. Training Configuration
Key hyperparameters and settings:
```
# Model Configuration
MAX_LENGTH = 256
MAX_GEN_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
PATIENCE = 7
GRADIENT_CLIP = 1.0
ACCUMULATION_STEPS = 4
```

#### 7. Implementation Notes
1. **Directory Structure**
- Automated creation of necessary directories
- Organized file structure for `logs`, `models`, and `data`
- Consistent path handling across different environments

2. **Error Handling**
- Added validation for data loading
- Implemented graceful failure handling
- Enganced error reporting and logging

3. **Performance Optimization**
- Implemented parallel data loading
- Added model optimization techniques
- Included progress tracking and visualization

4. **Code Organization**
- Modular class structure
- Clear seperation of concerns
- Improved maintability and readability

### Data Augmentation

The dataset is augmented using `QueryAugmenterNlpAug` class to increase lingustic diversity while preserving the semantic structure. The augmentation process includes:

#### 1. Initialization and Preprocessing

The system initializes with several crucial setup steps:

- Downloads required NLTK packages:
    - wordnet
    - averaged_perceptron_tagger_eng
- Initializes regex patterns for identifying:
    - 130-character hexadecimal values
    - Unix timestamps (10-digit numbers) 
    - Function patterns (e.g., "getCoordinate function") 
    - Directional patterns (e.g., "to/from/by [hex]")
    - Temporal patterns (e.g., "after/before [timestamp]")

#### 2. Preservation Rules:
The system maintains critical components through careful preservation rules:
- Special patterns are automatically added to stopwords:
    - Hexadecimal public key
    - Timestamps
    - Function patterns
    - Directional indicators for public keys (e.g., "to [hex]")
    - Temporal indicators with timestamps (e.g., "after [timestamp])

#### 3. Augmentation Techniques:

This system employs two distinct augmentation methods:

##### BERT Contextual Word Substitution

```python
naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    device=self.device,
    action="substitute",
    aug_p=0.3,
    stopwords=stopwords
)
```
- Replaces words with semantically similar alternatives
- 10% probability of word replacement
- Minimum one word augmentation
- Preserved stopwords and critical terms
- Examples:
    - Original: "Show two transactions from {hex} after {timestamp}"
    - Augmented: "Display two transactions from {hex} after {timestamp}"

##### BERT Contextual Word Substitution (BERT)

```python
naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    device='cuda',
    action="substitute",
    aug_p=0.1,
    aug_min=1,
    stopwords=stopwords
)
```
- Uses BERT model for context-aware replacements
- 30% substitution probability
- Maintains contextual relevance while preserving critical patterns
- Examples:
    - Original: "Get transactions to {hex} setup function"
    - Augmented: "Fetch records to {hex} setup function"
- Note: `WordNet-based synonym substitution` was considered but not implemented as it provides similar but less sophisticated replacements compared to BERT.

##### Random Word Swap

```python
naw.RandomWordAug(
    action="swap",
    aug_p=0.3,
    stopwords=additional_stopwords
)
```
- Creates syntactic variations through controlled word swapping
- 30% swap probability
- Uses additional stopwords to preserve extended patterns
- Examples:
    - Original: "Query recent transactions from {hex}"
    - Augmented: "Recent transactions query from {hex}"

![Data Augmentation](https://github.com/Winter-Zzzz/AI_dashboard/blob/main/image/data_augmentation_result.png?raw=true)

#### 4. Processing Pipeline

The augmentation process follows a structured pipeline:

##### Pre-processing

- Pattern identification and preservation
- Stopwords initialization for each augmentation method
- Device setup (CUDA if available, CPU otherwise)

##### Augmentation Process

- Maintains original input-output pairs
- Applies each augmentation method sequentially
- Generates on variation per method for each input
- Filters out augmentations containing `UNK` tokens
- Progress tracking with tqdm

#### Data Managment

- Saves results to JSON format maintaining input-output pair structure
- Includes comprehensive error handling and logging
- Provides absolute path references for data files

The augmentation system leverages BERT-based contextual substitution and controlled word swapping while preserving critical patterns to generate linguistically diverse but semantically consistent variations of the original queries.

### Fine Tuning
#### 1. Core Configuration
##### Model Settings
```python
@dataclass
class ModelConfig:
    MODEL_NAME: str = 't5-small'
    MAX_LENGTH: int = 256
    MAX_GEN_LENGTH: int = 256
    VOCAB_SIZE: int = 32100
    
    # Training Parameters
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 0.01
    NUM_EPOCHS: int = 5
    PATIENCE: int = 3
```

##### Directory Management
```python
def __post_init__(self):
    self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, 'models')
    self.BEST_MODEL_DIR = os.path.join(self.MODEL_DIR, 'best_model')
    self.FINETUNED_MODEL_DIR = os.path.join(self.MODEL_DIR, 'fine_tuned_model')
```
#### 2. Implementation Components
##### Model Initialization

```python
model_dir = os.path.join(project_root, 'models', 'best_model')
checkpoint_path = os.path.join(model_dir, "model_checkpoint.pt")

tokenizer = T5Tokenizer.from_pretrained(model_dir)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # Added weights_only parameter
model = T5ForConditionalGeneration(t5_config)
model.resize_token_embeddings(len(tokenizer))  # Added token embedding resize
model.load_state_dict(checkpoint['model_state_dict'])
```

#### 2. Training Infrastructure
##### Data Management
```python
train_dataset = QueryDataset(input_texts, output_texts, tokenizer, config.MAX_LENGTH)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config.BATCH_SIZE, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True if torch.cuda.is_available() else False  # Updated conditional pin_memory
)
```
##### Training Setup
```python
optimizer = AdamW(
    model.parameters(), 
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY, 
    eps=1e-8
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),  # Updated warmup steps calculation
    num_training_steps=total_steps
)

scaler = GradScaler()  # Added GradScaler initialization
```

##### Training Loop
```python
for epoch in range(start_epoch, start_epoch + config.NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    train_steps = 0  # Added step counter
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        with autocast(device_type='cuda'):
            output = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device), 
                labels=batch['labels'].to(device)
            )
            loss = output.loss / config.ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()

        if (i + 1) % config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * config.ACCUMULATION_STEPS
        train_steps += 1

        progress_bar.set_postfix({
            'loss': f'{loss.item() * config.ACCUMULATION_STEPS:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
```

#### 3. Key Optimizations
##### Memory Management
- Mixed precision training (FP16)
- Gradient acculuation (4 steps)
- Automatic GPU/CPU detection
```python
@property
def device_settings(self):
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_gpu': torch.cuda.device_count(),
        'fp16': self.FP16 and torch.cuda.is_available()
    }
```
##### Training Stability
- Gradient clipping (1.0)
- Early stopping (patience: 3)
- Learning rate warmup (10%)
```python
if (i + 1) % config.ACCUMULATION_STEPS == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
    scaler.step(optimizer)
```

##### Progress Tracking
```python
def update(self, epoch, train_loss):
    improved = train_loss < self.best_loss
    self.training_history['improvement'].append(improved)
    if improved:
        self.best_loss = train_loss
    self.plot_progress()
    self.save_status(epoch)
    return improved
```

#### 4. Results Handling
##### Checkpointing
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'tokenizer_vocab': tokenizer.get_vocab(),
    'tokenizer_special_tokens_map': tokenizer.special_tokens_map,
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'epoch': epoch,
    'loss': avg_train_loss,
}
```

##### Progress Visualization
```python
def plot_progress(self):
    plt.figure(figsize=(12, 6))
    plt.plot(self.training_history['epochs'], 
             self.training_history['train_loss'], 
             label='Training Loss', 
             marker='o')
    
    improved_epochs = [e for i, e in enumerate(self.training_history['epochs']) 
                      if self.training_history['improvement'][i]]
    improved_losses = [l for i, l in enumerate(self.training_history['train_loss']) 
                      if self.training_history['improvement'][i]]
    
    if improved_epochs:
        plt.scatter(improved_epochs, improved_losses, 
                   color='green', s=100, 
                   label='Improvement', 
                   zorder=5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
```

### Installation and Setup

Follow these steps to set up and run the project:

1. Download Required Model File: [models.safetensors](https://drive.google.com/file/d/1JC8L-BEQPzxPByrW-_M_1jrwgUfg-a-y/view?usp=drive_link)

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
### Initial Training Results
- Training Loss: 0.0099
- Validation Loss: 0.0070
- Best Loss: 0.00467
- Total Improvements: 19
- Final Epoch: 20

### Training Progress
The model showed consistent improvement throughout the training process:

- Training loss decreased significantly from 2.4846 to 0.0099
- Validation loss reduced from 0.9378 to 0.0070
- Achieved 19 performance improvements over 20 epochs
- Best model checkpoint saved at /content/AI_dashboard/ai/models/best_model

These results indicate that the model trained successfully with proper generalization without overfitting.

### Fine-tuning Analysis
After attempting model fine-tuning, we observed that the fine-tuned version did not provide meaningful improvements over the original model:

1. **Performance Metrics**
   - Inconsistent loss values during fine-tuning
   - Early stopping triggered after 7 epochs without improvement
   - Test generations showed degraded output quality

2. **Output Quality Issues**
   - Incorrect syntax in generated code
   - Inconsistent query handling
   - Format inconsistencies in outputs

Based on these results, we decided to proceed with the original pre-trained model, which demonstrated better stability and accuracy in query translation.

### Performance Analysis of Original Model

#### 1. Query Processing Accuracy
Strong performance across various query types:

1. **Basic Queries**
```
Input: "Get all most recent setPowerState txn"
Output: "txn.by_pk(-1).by_src_pk(-1).by_func_name('setPowerState').by_order(1).get_result(-1)"
```

2. **Complex Time-based Queries**
```
Input: "Appearance all before <time> 1794595202 </time>"
Output: "txn.by_pk(-1).by_src_pk(-1).by_func_name(-1).before('1794595202').by_order(0).get_result(-1)"
```

3. **Hex Address Handling**
```
Input: "Show all most recent to <hex> 8f25b3a... </hex>"
Output: "txn.by_pk('8f25b3a...').by_src_pk(-1).by_func_name(-1).by_order(1).get_result(-1)"
```

#### 2. Key Strengths
- Perfect handling of hex addresses
- Accurate time-based query processing
- Correct function name recognition
- Proper parameter ordering

## Conclusion

This paper presented AI_dashboard, an innovative solution that bridges the gap between natural language interaction and blockchain data analysis. Our implementation successfully demonstrates several key achievements while identifying areas for future enhancement.

### Key Achievements

1. **Natural Language Processing Integration**
   - Successfully implemented natural language query processing for blockchain data
   - Achieved high accuracy in query translation to executable code
   - Developed robust pattern recognition for complex queries

2. **Model Performance**
   - Achieved optimal performance with initial training
   - Successfully handled various query types including timestamps, function names, and hex addresses
   - Demonstrated stable performance across different query patterns

3. **Technical Implementation**
   - Built efficient data generation and augmentation pipeline
   - Implemented robust training infrastructure with optimized memory management
   - Developed comprehensive API server implementation

### System Benefits

1. **Accessibility**
   - Simplified blockchain data access for non-technical users
   - Eliminated need for specialized query language knowledge
   - Reduced learning curve for new team members

2. **Data Integrity**
   - Maintained data authenticity through direct blockchain integration
   - Eliminated intermediary systems that could compromise data
   - Preserved blockchain's inherent immutability

3. **Performance**
   - Achieved real-time query processing
   - Implemented efficient memory management
   - Demonstrated stable performance across various query types

### Future Work

1. **Model Enhancement**
   - Further expansion of the training dataset
   - Integration of more complex query patterns
   - Investigation of alternative fine-tuning approaches

2. **Feature Development**
   - Addition of visualization capabilities
   - Implementation of batch processing
   - Enhancement of query complexity handling

3. **System Optimization**
   - Further performance optimization for large-scale deployments
   - Enhancement of real-time processing capabilities
   - Implementation of advanced caching mechanisms

The project successfully demonstrates the potential of combining AI with blockchain technology to create accessible and efficient data analysis tools. The implemented solution provides a solid foundation for both academic research and practical applications in the blockchain industry, while offering clear pathways for future enhancements and optimizations.