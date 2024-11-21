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

tbd ...

## Methodology
### Natural Language Command Augmentation with GPT API

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

## Evaluation

tbd ...

## Conclusion

tbd ...
