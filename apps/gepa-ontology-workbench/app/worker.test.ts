import { describe, expect, it, vi } from "vitest";

vi.mock("vinext/server/image-optimization", () => ({
  DEFAULT_DEVICE_SIZES: [],
  DEFAULT_IMAGE_SIZES: [],
  handleImageOptimization: async (
    _request: Request,
    options: {
      transformImage: (
        body: ReadableStream,
        options: { width: number; format: string; quality: number },
      ) => Promise<Response>;
    },
  ) => options.transformImage(
    new ReadableStream(),
    { width: 64, format: "webp", quality: 80 },
  ),
}));

vi.mock("vinext/server/app-router-entry", () => ({
  default: { fetch: vi.fn() },
}));

import worker from "../worker";

describe("ontology workbench worker", () => {
  it("returns a defined gateway error when image transformation fails", async () => {
    const response = await worker.fetch(
      new Request("https://workbench.example/_vinext/image"),
      {
        ASSETS: { fetch: vi.fn() },
        IMAGES: {
          input: () => ({
            transform: () => ({
              output: () => Promise.reject(new Error("transform unavailable")),
            }),
          }),
        },
      } as never,
      {} as never,
    );

    expect(response.status).toBe(502);
    expect(await response.text()).toBe("Image transformation failed.");
  });
});
