FROM lmsysorg/sglang:latest

# Expose the port
EXPOSE 30000

# Set the entrypoint to launch the server
# Note: HF_TOKEN should be passed as an env var at runtime
ENTRYPOINT ["python3", "-m", "sglang.launch_server", "--host", "0.0.0.0", "--port", "30000"]
CMD ["--model-path", "meta-llama/Llama-3.1-8B-Instruct"]