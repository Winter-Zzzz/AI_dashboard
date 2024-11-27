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
1. `generated_dataset.json` : Primary dataset containing 50 pairs of:
    - Natural language queries for transaction filtering
    - Corresponding Python code implementations
    - Queries include various conditions such as:
        - Public key filtering (Both source and destination)
        - Timestamp-based filtering
        - Function name filtering (setup, on, off)
        - Sort orders (earliest, latest, most recent, oldest)
        - Result count limitations

2. `augmented_dataset.json` : An enhanced version of the generated dataset created through NLP augmentation techniques to increase data diversity while maintaining query-code consistency.

The detailed generation and augmentation processes are described in the Methodology section.

## Methodology
### Dataset Generation
We implemented a `TransactionFilterDatasetGenerator` class to create diverse yet structurally consistent query-code paris. The generator uses the following components:

    ```python
    commands = ['List']
    sort_orders = ['earliest', 'recent']
    functions = ['setup function', 'on function', 'off function']
    ```
The generation process involves:

1. Query Generation:
    - Randomly selects a commmand, sort_order, functions, pk, src_pk, timestamp, and count.
        ```python
        return f"{command} {count if count else ''} {sort_order if sort_order else ''} {condition}".strip()
        ```
    - Applies random combinations of filtering conditions:
        - Public key filtering (to/from address)
        - Function name filtering
        - Timestamp filtering
            ```python
            # random pk
            if random.choice([True, False]):  
                address = self.random_pk()
                conditions.append(f"to {address}")
            
            # random src_pk
            if random.choice([True, False]):  
                address = self.random_pk()
                conditions.append(f"from {address}")

            # random func_name
            if random.choice([True, False]): 
                func = random.choice(self.functions)
                conditions.append(f"{func}")

            # random timestamp 
            if random.choice([True, False]):  
                timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"])
                conditions.append(timestamp)
            ```
    - Ensures at least one filter condition is always included
        ```python
            # At least one condition is needed
                if not conditions:
                    fallback_filter = random.choice(['to', 'from', 'func', 'timestamp'])
                    if fallback_filter == 'to':
                        address = self.random_pk()
                        conditions.append(f"to {address}")
                    elif fallback_filter == 'from':
                        address = self.random_pk()
                        conditions.append(f"from {address}")
                    elif fallback_filter == 'func':
                        func = random.choice(self.functions)
                        conditions.append(f"{func}")
                    elif fallback_filter == 'timestamp':
                        timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"])
                        conditions.append(timestamp)
            ```

2. Code Generation:
    - Constructs filter chain based on the query conditions
        ```python
        filter_chain = ""

        if "to " in input_text:
            pk = input_text.split("to ")[-1].split()[0]
            filter_chain += f".by_pk('{pk}')"
            
        if "from " in input_text:
            src_pk = input_text.split("from ")[-1].split()[0]
            filter_chain += f".by_src_pk('{src_pk}')"

        if any(func in input_text for func in self.functions):
            func_name = next(func for func in self.functions if func in input_text)
            filter_chain += f".by_func_name('{func_name}')"

        if "after " in input_text or "before " in input_text:
            timestamp = input_text.split("after ")[-1].split()[0] if "after " in input_text else input_text.split("before ")[-1].split()[0]
            filter_chain += f".by_timestamp('{timestamp}')"

        if "earliest" in input_text or "oldest" in input_text:
            filter_chain += ".sort()"

        elif "latest" in input_text or "most recent" in input_text:
            filter_chain += ".sort(reverse=True)"
        
        if re.findall(r'\b(?![0-9a-fA-F]{130}\b)(?!\d{10}\b)\d+\b', input_text):
            count = int(re.findall(r'\b(?![0-9a-fA-F]{130}\b)(?!\d{10}\b)\d+\b', input_text)[0])
            filter_chain += f".get_result()[:{count}]"

        else:
            filter_chain += ".get_result()"

Examples of a `generated_dataset` pair:

```json
{
    "dataset": [
        {
            "input": "Give  earliest from 4d494b88117d00842c348de49a8739f1c28a28a26c0da53b2bdfe6fc4378048170fa9b1bf455cf996e08fd52bfc46358666251dabb0c8f27796397a281226e0816",
            "output": "print(TransactionFilter(data).by_src_pk('4d494b88117d00842c348de49a8739f1c28a28a26c0da53b2bdfe6fc4378048170fa9b1bf455cf996e08fd52bfc46358666251dabb0c8f27796397a281226e0816').sort().get_result())"
        },
        {
            "input": "Retrieve 1 oldest to 53e726abfeefa4dc8dbd1afb1d3142b22361ffd5f42bb3277a10f5ce3652cf2b82ac39a9f9321492be1c4e091efc92dd67540175b6272d3c824a8c8c544604f6aa",
            "output": "print(TransactionFilter(data).by_pk('53e726abfeefa4dc8dbd1afb1d3142b22361ffd5f42bb3277a10f5ce3652cf2b82ac39a9f9321492be1c4e091efc92dd67540175b6272d3c824a8c8c544604f6aa').sort().get_result()[:1])"
        },
        {
            "input": "Give 7  to 97c8c74fa1b10823ce5c55f3ff43a886d4101cc7d03e9630e01715e7ff17f6e86e0c248ffea16a24d20cffa68b293f26ec9dc2ca36ee70cffa339f91a1c871e332 from 259789c392669608e7f366f125746f46cd0107e4b440c56c4c430e74fe1cd5528fa54e43fbbce2d2acb1e24482e09f7afe815c558f1d1b16c906fffb6adb61f4cb off function",
            "output": "print(TransactionFilter(data).by_pk('97c8c74fa1b10823ce5c55f3ff43a886d4101cc7d03e9630e01715e7ff17f6e86e0c248ffea16a24d20cffa68b293f26ec9dc2ca36ee70cffa339f91a1c871e332').by_src_pk('259789c392669608e7f366f125746f46cd0107e4b440c56c4c430e74fe1cd5528fa54e43fbbce2d2acb1e24482e09f7afe815c558f1d1b16c906fffb6adb61f4cb').by_func_name('off function').get_result()[:7])"
        }
    ]
}
```

### Data Augmentation
#### Overview
The data augmentation process in this project enhances the diversity and robustness of training data by applying advanced NLP augmentation techniques. The goal is to simulate natural variations in query expressions while maintaining semantic consistency with the corresponding code outputs. This ensures that models trained on the data generalize better to unseen inputs.

#### Objectives
1. Increase Data Diversity: By introducing natural variations to input queries, the model becomes more resilient to changes in phrasing, structure, and synonyms.
2. Preserve Semantic Integrity: While inputs are modified, the augmented outputs maintain functional correctness by aligning with the input semantics.
3. Simulate Real-world Inputs: The process mimics variations in human queries, preparing the model to handle diverse language patterns.

#### Augmentation Techniques
The augmentation process leverages NLP Augmentation (NlpAug) with the following techniques:

1. Synonym Replacement
    - Uses `naw.SynonymAug` to replace words in the input text with their synonyms from WordNet.
    - Preserves critical keywords (e.g., function names, transaction terms, and specific hex patterns) by dynamically generating stopwords.
    - Example: "show three transactions" → "display three transactions."

2. Contextual Word Substitution
    - Uses `naw.ContextualWordEmbsAug` with a pre-trained bert-base-uncased model to replace words based on context.
    - Ensures replacements fit naturally within the query's structure.
    - Example: "retrieve the latest transactions" → "fetch the latest transactions."

3. Contextual Word Insertion
    - Inserts contextually appropriate words into the input query, enriching its phrasing while maintaining meaning.
    - Example: "list transactions before timestamp" → "list all transactions before the given timestamp."

4. Pattern-based Variations
    - Template-based variations are applied to generate diverse queries while preserving the intent.
    - Variations include:
        - Adding or altering temporal expressions (e.g., "before timestamp").
        - Incorporating PK (primary key) and hexadecimal representations.
    - Example:
    Template: "{action} {number} {trans} {time_rel} {time_ctx}"
    - Generated Query: "show three transactions after this timestamp."

#### Validation and Consistency
To ensure the quality and relevance of augmented data:

1. Keyword Preservation: Critical keywords (e.g., function names, PKs, timestamps) are detected and preserved.
2. Structural Mapping: Hexadecimal values and timestamps are dynamically replaced while maintaining mapping consistency.
3. Output Validation: Augmented outputs are tested against original outputs to ensure functional equivalence.

#### Data Integrity
- Redundant or invalid augmented pairs are filtered out.
- Only unique, validated input-output pairs are included in the final dataset.

---


To enhance the diversity and flexibility of natural language commands, we use OpenAI's GPT API for data augmentation. This approach generates alternative expressions of input commands, allowing the system to generalize and adapt to various linguistic structures. Below is the detailed methodology:

1. Command Transformation Techniques:
    - Synonym Replacement: Words or phrases are replaced with their synonyms while preserving the original meaning.*Example: "Retrieve" → "Fetch"*
    - Sentence Reordering: The structure of the command is rearranged without altering its intent.*Example: "Show the latest 5 transactions from Public Key 0000" → "From Public Key 0000, show the latest 5 transactions."*
    - Active/Passive Voice Conversion: Commands are converted between active and passive voices.*Example: "Retrieve data from Public Key 0000" → "Data should be retrieved from Public Key 0000."*

2. Preserving Semantic Integrity:
    - GPT is prompted to rephrase commands while maintaining the core meaning.
    - Variations may include additional context, slight expansions, or simplified expressions.

3. Stylistic Variations:
    - Formal and Informal Styles: Commands are adjusted to reflect different tones or levels of formality.*Example: "Fetch the data now, please" → "Retrieve the data immediately."*
    - Colloquial Expressions: Commands are rephrased into conversational forms for casual usage.*Example: "Public Key 0000 transactions, show me 5" → "Hey, can you show 5 transactions from Public Key 0000?"*

4. Text Simplification and Expansion:
    - Commands can be simplified into concise phrases or expanded with additional details.*Example (Simplified): "Show 5 recent transactions."Example (Expanded): "Retrieve the 5 most recent transactions associated with Public Key 0000 in the system."*

5. Context-Specific Vocabulary:
    - Augmented commands include domain-specific terminologies, such as blockchain-related terms.*Example: "Get the last 5 transactions" → "Fetch the 5 latest blocks linked to Public Key 0000."*

6. Implementation:
    - We use the GPT API’s `text-davinci-003` model for augmentation.
    - A predefined prompt instructs GPT to generate a fixed number of alternative expressions.
    - The results are filtered and tokenized for use in model training and validation.

7. Benefits:
    - Increases the dataset size for training without manual effort.
    - Enhances the model's ability to understand varied user inputs.
    - Prepares the system for real-world applications by accommodating different linguistic nuances.

### Model Fine-tuning process

1. Pre-training Model Selection
    - Base Model: T5-base
    - Rationale: Text-to-text framework 

2. Dataset Preparation
    - Source- Source: "translate to query: {command}"
    - Target: Blockchain query
    - Format: instruction-input-output pairs    

3. Fine-tuning Implementation
    ```python
    # Initiate tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Preprocess Data
    def preprocess_data(examples):
        inputs = ["translate to query: " + text for text in examples["command"]]
        
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = tokenizer(
            examples["query"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

    # Training configurations
    training_args = TrainingArguments(
        output_dir="./t5-blockchain-query",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_steps=100,
    )

    # Initialize trainer with model and training configurations
    trainer = Trainer(
        model=model,                
        args=training_args,         
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer,
    )

    # Start the fine-tuning process
    trainer.train()
    ```

The training process will:
- Update the model weights using the specified learning rate
- Log training progress at regular intervals
- Save model checkpoints according to the specified strategy
- Monitor training metrics for optimization

## Evaluation

tbd ...

## Conclusion

tbd ...
