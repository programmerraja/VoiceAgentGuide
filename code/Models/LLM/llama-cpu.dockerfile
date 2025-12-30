FROM ollama/ollama

# Expose the default Ollama port
EXPOSE 11434

# Start the server
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]

