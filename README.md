# Testing Report
To thoroughly test the DataXID SDK, I designed an automated testing harness using LLM-driven agents (via **Ollama** llama3.1-8b). I specifically designed these agents to evaluate the SDK's "Statistical Integrity" and "Behavioral Regression" by generating adversarial inputs and edge-case configurations. During the development of this testing suite, I made some architectural decisions regarding both the testing process and the SDK’s behaviour.

Firstly I would like to mention that I created 3 different agent: Edge-Case Junior, Developer Experience Critic Elf, Functional Specialist Goblin. They are doing what they naming.

In some cases agents **stuck getting same errors** so to prevent the agents from getting stuck in a loop of triggering the exact same error I implemented **“Exploration” and “Entropy” mechanism**s. This forces the agents to constantly shift their attack vectors across different functional domains but this sometimes worked sometimes didn’t.

Other observation of my is realizing that **feeding all available SDK parameters** to every agent distracted their focus. So I strictly scoped configuration parameters to match each agent’s specif role. Also I provided a pool of all config parameters so that they can fetch what they need.

Beginning of the projects **all the found bugs actually not bugs that related to the SDK**. When I realized this then I tried to filter out several false positives where the testing framework failed. Some of them are **JSON parse errors**; **context window poisoning** which is initially I fed the entire test history back into the model but then realized that this overloaded the context window and cause hallucinations; **global test history contamination** which means firstly as I mentionad I used global test history but I also figure out that when running different agents all their results collected that global test collection, this is not what we want because agents have different roles they need to focus. If they see irrelavant test outputs then their focus will be distracted so I made agent history instead of global history. One another is the agents frequently realized that passing an **empty data_dict** was the easiest way to trigger an error and they all make same things after that. This is also not we want from agents. We want them find different bugs not always same.

What I did at sometime is the try to **threading** and **multiprocessing** the **synthesize()** function. Because sometimes it runs 5 minutes, 10 minutes or more. So I tried to implement a timeout mechanism using _concurrent.futures.ThreadPoolExecuter_ with _MAX_WAIT_SECONDS_ threshold. However even when the main thread throws a _TimeoutError_ the underlying _dataxid.synthesize_  does not terminate, forcing Python to hang while waiting for the resource to free up. I explored using multiprocessing to aggressively kill the process, but ultimately removed it as it introduced too much structural complexity for this testing project.

# Bug Reports
## Report 1
### Title
Crash on Adversarial Input with Large Batch Size and Unhashable Type Error
### Severity Score
Critical(10) – SDK not working
### Description
During an adversarial test, the DataXID SDK crashed when processing input data containing extremely large feature values. The error was triggered by attempting to hash a list object as a key in a dictionary, indicating a failure of the internal data structure management.
### Expected Behavior
The SDK should validate the column data types prior to internal processing. If lists or unhashable types are unsupported for standard tabular generation, it should raise a clear, user-friendly ValueError or DataXIDValidationError explaining that nested lists are not supported.
### Actual Behavior
Crashes with the raw exception: TypeError: unhashable type: 'list'
### Steps to Reproduce
1. Initialize SDK configuration with:
```py
config_params = {'embedding_dim': 256, 'val_split': 0.2, 'learning_rate': 1e-05}
```
2. Prepare dataset with structure:
```py
data_dict = {'features': [[1000000, 2000000], [3000000, 4000000]], 
             'labels': [[1, 2], [3, 4]]}
```
3. Execute process with `n_samples = 1024`
4. Observe the crash.

## Report 2
### Title
DataXID SDK Crashes When Processing High-Dimensional Sensor Data with Privacy Noise
### Severity Score
Major(7) - Function works but output/execution breaks under specific valid configurations
### Description
During an adversarial test, the DataXID SDK crashed when processing high-dimensional sensor data with increasing levels of privacy noise. This is a failure because the SDK is expected to maintain performance and handle edge-cases without crashing.
### Expected Behavior
The SDK should have handled the high-dimensional sensor data and increasing levels of privacy noise gracefully without crashing. Instead, it crashed with an exception indicating that only integer tensors of a single element can be converted to an index.
### Actual Behaviour
Crashes with exception: `only integer tensors of a single element can be converted to an index`
### Steps to Reproduce
1. Initialize SDK configuration with: `config_params = {'seed': 12345, 'privacy_enabled': True, 'privacy_noise': [0.05, 0.1, 0.2], 'encoding_types': {'timestamp': 'TABULAR_DATETIME', 'temperature': 'TABULAR_NUMERIC_AUTO', 'humidity': 'TABULAR_CATEGORICAL'}, 'model_size': 'large'}`
2. Prepare dataset with structure: `data_dict = {'timestamp': [1643723400, 1643723405, 1643723410], 'temperature': [20.12345, 20.45678, 20.90123], 'humidity': [60.12345, 61.23456, 62.34567], 'acceleration_x': [-1.23456, -2.34567, -3.45678]}`
3. Execute process with `n_samples = 3000`
4. Observe the crash.

## Report 3
### Title
Poor DX on Type Mismatches in ModelConfig
### Severity Score
DX Feedback (4) - Not a bug, but improvement suggestion for developer experience.
### Description
If a developer accidentally passes a string instead of an integer for configuration parameters (e.g., '1024' instead of 1024 for embedding_dim), the ModelConfig object accepts it without complaining. The error only surfaces much deeper in the stack during the PyTorch empty() tensor allocation, resulting in a highly confusing, low-level error trace that wastes developer debugging time.
### Expected Behavior
_dataxid.ModelConfig(...)_ should strictly enforce type hinting via Pydantic or a similar validation layer. If an invalid type is passed, it should fail immediately upon configuration initialization with a clear message like _ValueError: embedding_dim must be an integer_.
### Actual Behaviour
The config initializes successfully, but execution crashes deep in the stack with: _TypeError: empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=torch.device)..._
### Steps to Reproduce
1. Initialize SDK configuration with: `config_params = {'encoding_types': {'text': 'utf8', 'label': 'int'}, 'val_split': 0.2, 'batch_size': 128, 'model_size': 'large', 'embedding_dim': '1024'}`
2. Prepare dataset with structure: `data_dict = {'col1': ['embedding_dim', 'batch_size'], 'col2': [1, 128]}`
3. Execute process with `n_samples = 1000`
4. Observe the crash.

## Report 4
### Title
Training Crash due to Gradient Accumulation & Batch Size Incompatibility
### Severity Score
Major (7) - Function fails under valid but edge-case configurations.
### Description
During an adversarial test, the DataXID SDK crashed when attempting to train with incompatible configuration settings. This occurred because the 'accumulation_steps' parameter was misinterpreted as requiring a large batch size value instead of enabling gradient accumulation.
Expected Behavior: The SDK should handle the edge-case where 'accumulation_steps' is enabled and automatically adjust batch size to a large value (e.g., 1<<25) without crashing or producing an incorrect error message.
### Actual Behavior
Crashes with exception: `Training failed: training_failed`
### Steps to Reproduce
1. Initialize SDK configuration with: `config_params = {'encoding_types': {'default': 'utf-8'}, 'val_split': 0.1, 'batch_size': 100, 'accumulation_steps': 5}`
2. Prepare dataset with structure: `data_dict = {'col1': [100], 'col2': [5]}`
3. Execute process with `n_samples = 128`
4. Observe the crash

## Report 5
### Title
Type Mismatch and Division Crash in batch_size Handling
### Severity Score
Minor (3) - Small issue, likely due to misleading internal type expectations.
### Description
The DataXID SDK crashed when processing a dataset with an integer value assigned to the `batch_size` parameter in the `config_params`. This failure occurs because the SDK incorrectly expects a string input for this parameter, leading to a silent ignore and subsequent calculation errors. The underlying mechanism that broke is the incorrect type handling for the `batch_size` parameter.
### Expected Behavior
The SDK should handle this edge-case by either:
	Raising a clear error message when an integer value is assigned to the `batch_size` parameter.
	Providing additional checks or documentation to ensure correct input type usage, preventing silent ignores and calculation errors
### Actual Behavior
Crashes with exception: `unsupported operand type(s) for /: 'str' and 'int'`
###Steps to Reproduce
1. Initialize SDK configuration with: `config_params = {'encoding_types': {'type1': 'category', 'type2': 'numerical'}, 'val_split': 0.8, 'batch_size': '200', 'model_size': 'small', 'embedding_dim': 128}`
2. Prepare dataset with structure: `data_dict = {'col1': ['example1', 'example2'], 'col2': ['example3', 'example4']}`
3. Execute process with `n_samples = 100`
4. Observe the crash
