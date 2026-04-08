# Checked Features
- `python setup.py --app-home E:/arignan`: Works as expected
- `arignan detete load-20260408073121-0c125daf`: Works as expecteda
- `arignan delete --hat SNNs`: Works as expected
- `arignan save-session 8apr`: Works as expected. 
- `arignan load-session 8apr`: Works as expected.
- `arignan load "JEPA\V-JEPA2.1.pdf" --hat "SNNs" --debug`: Works as expected.

# Known Issues
- `arignan ask "what is the full form of jepa"`: Is not hitting LLM
- Ask commands are printing too much verbose instead of just giving a quick answer
