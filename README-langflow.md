# SCK Langflow Workbench & MCP Integration

The Langflow workbench Compose file has been moved to `langflow/compose.yml` because it is **only** for
running the external Langflow UI (workbench) and optional helper services. Our own SCK AI container
image (MCP + service APIs) is built independently via the project Dockerfile.

## Quick Start

1. **Change directory to the compose file**:
   ```powershell
   cd .\langflow
   ```

2. **(Optional) Create data directories** (if you plan to persist / extend beyond in-memory SQLite):
   ```powershell
   ..\setup-langflow.ps1
   ```

3. **Start services** (Langflow UI + MCP HTTP bridge + SCK tools server):
   ```powershell
   docker compose up -d
   # or: docker-compose up -d (legacy syntax)
   ```

4. **Access Langflow Workbench**:
   - URL: http://localhost:7860
   - Username: admin
   - Password: admin123

5. **MCP HTTP Bridge** (wraps the stdio MCP server and exposes simple HTTP endpoints):
   - Base URL: http://localhost:8001
   - Endpoints: `/`, `/tools`, `/search-documentation`, `/search-codebase`, `/validate-cloudformation`

6. **SCK Tools Server** (direct lightweight helpers if you don't need MCP JSON-RPC):
   - Base URL: http://localhost:8002
   - Endpoints: `/search-docs`, `/search-code`, `/architecture`

7. **Import SCK Flows**:
   - In the Langflow UI, import: `sck-documentation-chat.json`, `sck-mcp-chat.json`, etc.
   - Add your OpenAI API key / providers to the relevant nodes.

## Commands (from `langflow/` directory)

- **Start all**: `docker compose up -d`
- **Stop**: `docker compose down`
- **Logs (all)**: `docker compose logs -f`
- **Logs (langflow only)**: `docker compose logs -f langflow`
- **Rebuild bridge/tools**: `docker compose build --no-cache mcp-http sck-tools`
- **Restart a service**: `docker compose restart mcp-http`
- **Update Langflow image**: `docker compose pull langflow && docker compose up -d`

## Data Persistence

The default compose uses an in-container SQLite DB (`/tmp/langflow.db`). For persistent data:

1. Add a volume mount in `langflow/compose.yml` under `langflow`:
    ```yaml
    volumes:
       - ./langflow-data:/data
    environment:
       - LANGFLOW_DATABASE_URL=sqlite:////data/langflow.db
    ```
2. (Optional) Switch to Postgres by uncommenting / adapting a `postgres` service (see earlier version history if needed).

## Troubleshooting

| Symptom | Check | Fix |
|---------|-------|-----|
| Langflow UI not loading | `docker compose ps` | Ensure port 7860 free / restart service |
| MCP bridge 500 errors | `docker compose logs mcp-http` | Verify `core_ai.mcp_server` imports succeed |
| Tools server missing flows | Mount path | Confirm `langflow/*` JSON flows exist |
| Flows can't call HTTP bridge | Network | All services share `sck-ai-network` by default |

If "Create a flow" is disabled after a cold start, restart just the Langflow container:
```powershell
docker compose restart langflow
```

## Linking Langflow Nodes to MCP HTTP Bridge

Use an HTTP request component in Langflow pointing to `http://host.docker.internal:8001/<endpoint>` on macOS/Windows Docker Desktop
or `http://mcp-http:8001/<endpoint>` from another service inside the same compose network.

Examples:
```text
GET  http://mcp-http:8001/tools
POST http://mcp-http:8001/search-documentation (JSON body: {"query": "deployment"})
```

For local host machine curl tests:
```powershell
curl http://localhost:8001/tools
curl -X POST "http://localhost:8001/search-codebase?query=MagicS3Bucket"
```