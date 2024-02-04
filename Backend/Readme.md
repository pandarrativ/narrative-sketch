To Enable Backend server, you need to 
1. use server with GPU (>12GB memory) to serve the languate model: python GPT2_server.py 
2. override the ip address of the GPT server in source file app_run.py: override the variable GPT_url at teh beginning
3. run python app_run.py
If you want to clear the user's history, just run: python cleanGarbage.py