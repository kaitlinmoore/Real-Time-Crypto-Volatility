I did make use of generative AI for this project. I utilized both Claude and ChatGPT as well as coding suggestions as I typed in my IDE (PyCharm.)

**Documents**

Prompt (summary): "Review this descriptive paragraph and outline.  Do not change the paragraph, but please incorporate the outline into a one-page scoping brief."
Used in: scoping_brief.md
Verification: I reviewed and edited the response.

Prompt (summary): "Attached are my features and a description of what each class of features does. Please format the infromation into a feature spec markdown document."
Used in: feature_spec.md
Verification: I reviewed and edited the response.

Prompt (summary): "Please review this drafted model card and provide feedback,"
Used in: model_card.md
Verification: I reviewed the feedback and incorporated some elements and rearranged.

Prompt (summary): "Please review the attached code files, project description, and requirements.txt and draft the README file for my GitHub repository."
Used in: README.md
Verification: I reviewed, heavily cut, and edited the response.


**Code**

Prompt (summary): "Here is my current code. I would like to improve performance time. Can you make optimization suggestions?"
Used in: replay.py, featurizer.py
Verification: I reviewed, selected options for testing, tested, edited, and implemented what was useful.

Prompt (summary): "Please prove the general code outline for ____"
Used in: train.py, tune.py, train_new.py, ws_ingest.py, infer.py
Verification: I used the skeleton code to build my drafts, tested, and then presented for additional feedback.

Prompt (summary): "I want these arguments in my function. Please format them in the appropriate python."
Used in: add_labels_perproduct.py
Verification: I reviewed, tested, and edited the responses.

Prompt (summary): "I am getting this error when I run the attached code. Can you help me debug?"
Used in: featurizer.py, train.py, eda.ipynb, ws_ingest.py, config.yaml
Verification: I reviewed, tested, went back if there were still errors, and edited.

Prompt (summary): "Please review my EDA notebook and make recommendations. It is attached."
Used in: eda.ipynb
Verification: I reviewed, considered, and selected changes to make.

Prompt (summary): "I have written this utilities file with some functions that I may need repeatedly. Do you have other recommendations that will be useful for this pipeline?"
Used in: utilities.py
Verification: I reviewed, considered, and selected changes to incorporate, though wound up not using many of them.

Prompt (summary): "Please draft the logging functions for this code."
Used in: train.py, train_new.py, ws_ingest
Verification: I reviewed, tested, changed a lot of the formatting.

Prompt (summary): "The tutorials I watched on evidently are from an older version. A lot of the API has changed. Can you help me update my attached script to follow the 0.7.16 standards?"
Used in: generate_evidently_report_update.py
Verification: I reviewed, tested, and incorporated edits.

Prompt (summary): "Here are two versions of my code. I think I fixed a bug in the first one. Can you compare them and give feedback?"
Used in: featurizer_l2_optimized.py, add_labels_fix.py
Verification: I read feedback, cross-referenced with other research, tested, and made some additional tweaks to my edits to omprove performance.

Prompt (summary): "I have not used FastAPI before. Please check this code for errors, and provide a clear explanation for any that are found."
Used in: main.py
Verification: I read feedback, cross-referenced with other research, tested, and edited.
