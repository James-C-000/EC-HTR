rem SSH tunnel for production.py: local port 8000 -> vLLM on the rented instance. Reconnects 10 s after a drop.
rem Fill in the three values from the instance's SSH command in the vast.ai dashboard.
set HOST=INSTANCE_IP
set SSH_PORT=INSTANCE_SSH_PORT
rem The port vLLM listens on inside the instance (18000 in the October 2026 production run)
set VLLM_PORT=18000

:loop
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 8000:127.0.0.1:%VLLM_PORT% -p %SSH_PORT% root@%HOST%

timeout /t 10
goto loop
