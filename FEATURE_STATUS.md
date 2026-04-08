# Checked Features
- `python setup.py --app-home E:/arignan`: Works as expected
- `arignan detete load-20260408073121-0c125daf`: Works as expecteda
- `arignan ask "what is the full form of jepa" --debug`: Works as expected
- `arignan ask "what is the full form of jepa"`: Works as expected

# Features with Issues
- `arignan load "JEPA\V-JEPA2.1.pdf" --hat "SNNs" --debug`: 
    - Quality of map.md and global_map.md ad too low
    - Summary markdown is garbled with no usable or readable imformation
- `arignan save-session 8apr`: The saves session has empty turns and context. Also saves the context in terminal's directory instead of app-home/sessions. 

# Potential Improvements
- Chunking token size can be increased
- Prompt given to LLM in ask command can be improved
- Prompts given to generate Knowledge-base can be improved
- Template examples for how to structure knowledge base markdowns and map markdowns can be given to the LLM