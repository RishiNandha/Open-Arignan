# Checked Features
- `python setup.py --app-home E:/arignan`: Works as expected
- `python setup.py --app-home E:/arignan --lightweight`: Supported
- `arignan detete load-20260408073121-0c125daf`: Works as expecteda
- `arignan delete --hat SNNs`: Works as expected
- `arignan -gui`: Implemented
- `arignan save-session 8apr`: Works as expected. 
- `arignan load-session 8apr`: Works as expected.
- `arignan load "JEPA\V-JEPA2.1.pdf" --hat "SNNs" --debug`: Works as expected.
- `arignan ask "what is the full form of jepa"`: Works as expected

# Known Issues
1. Answers are just rephrasing the given context, which adds no value for having the LLM. Prompting style needs changes
2. Summary markdown can be more strictly wiki-styled with cross-lines (as described by A Karpathy recently)
3. Topic grouping is still not very intelligent. Some topics are not getting grouped even when similar
4. Segmentation of a large book into multiple sections hasn't been tested yet
