To run a trained model on a DepthAI module, only 3 files are needed: the `.blob` file and the two `.json` files.
The OpenVINO 20.01 Intermediate Representation files `.bin` and `.xml` are provided for convenience.

Copy the folder in your local DepthAI folder, under resources/nn. 
Then run a terminal in the DepthAI folder and type  `python3 test.py -cnn medmask` (for the medmask model).
