import { apiRequest } from "../lib/api";
import type { WorkspaceSummary } from "./types";

export async function getWorkspaces(
  signal?: AbortSignal,
): Promise<WorkspaceSummary[]> {
  return apiRequest<WorkspaceSummary[]>("/workspaces", {
    method: "GET",
    signal,
  });
}