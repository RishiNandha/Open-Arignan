# Checked Features
- `python setup.py --app-home E:/arignan`: Works as expected
- `arignan detete load-20260408073121-0c125daf`: Works as expecteda
- `arignan delete --hat SNNs`: Works as expected
- `arignan save-session 8apr`: Works as expected. 
- `arignan load-session 8apr`: Works as expected.
- `arignan load "JEPA\V-JEPA2.1.pdf" --hat "SNNs" --debug`: Works as expected.
- `arignan ask "what is the full form of jepa"`: Works as expected

# Improvements
- Answering quality can be improved with better more structured prompt given to the LLM instead of just plainly adding context and question
- Summary markdown quality can be improved with better examples of how a wiki-style knowledge base should look
