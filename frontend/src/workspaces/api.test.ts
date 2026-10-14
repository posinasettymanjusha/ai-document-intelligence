
import { describe, expect, it, vi } from "vitest";
import { getWorkspaces } from "./api";
import { apiRequest } from "../lib/api";
import type { WorkspaceSummary } from "./types";

vi.mock("../lib/api", () => ({
  apiRequest: vi.fn(),
}));

describe("getWorkspaces", () => {
  it("requests and returns the workspace list", async () => {
    const workspaces: WorkspaceSummary[] = [
      {
        id: "workspace-1",
        name: "My Workspace",
        role: "owner",
      },
    ];

    vi.mocked(apiRequest).mockResolvedValue(workspaces);

    const result = await getWorkspaces();

    expect(apiRequest).toHaveBeenCalledWith("/workspaces", {
      method: "GET",
      signal: undefined,
    });

    expect(result).toEqual(workspaces);
  });

  it("propagates API errors", async () => {
    const error = new Error("Request failed");

    vi.mocked(apiRequest).mockRejectedValueOnce(error);

    await expect(getWorkspaces()).rejects.toThrow("Request failed");
  });
});
