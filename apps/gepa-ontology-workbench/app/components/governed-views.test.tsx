/** @vitest-environment jsdom */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { assessments, invariants, ontologyNodes, ontologyRelations } from "../data/ontology";
import type { Assessment, OntologyNode } from "../lib/ontology-types";
import { AssessView } from "./AssessView";
import { ConceptDetail } from "./ConceptDetail";
import { ImproveView } from "./ImproveView";
import { InvariantsView } from "./InvariantsView";
import { Workbench } from "./Workbench";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const renderImprove = (overrides: Partial<React.ComponentProps<typeof ImproveView>> = {}) => render(
  <ImproveView
    selectedId="failure:goal_fixation"
    nodes={ontologyNodes}
    relations={ontologyRelations}
    assessments={assessments}
    invariants={invariants}
    onOpenInvariant={() => undefined}
    {...overrides}
  />,
);

describe("governed workbench views", () => {
  it("uses verified canonical wording only after the content digest passes", async () => {
    render(<Workbench />);

    const digestStatus = screen.getByRole("status", { name: "Canonical ontology digest status" });
    expect(digestStatus).toHaveTextContent("Verifying canonical digest");
    await waitFor(() => expect(digestStatus).toHaveTextContent("Verified canonical"));
  });

  it("preserves unavailable evidence and exposes evidence-quality warnings", () => {
    const assessment: Assessment = {
      ...assessments[0],
      support: { kind: "confidence" },
      opposition: { estimate: 0.8, kind: "confidence" },
      provenance: [],
    };
    const nodes: readonly OntologyNode[] = ontologyNodes.map((node) => node.id === assessment.subject
      ? { ...node, maturity: { ...node.maturity, dataset_backed: false } }
      : node);

    render(<AssessView assessments={[assessment]} nodes={nodes} />);

    expect(screen.getByText("Unavailable")).toBeVisible();
    expect(screen.getByText("Evidence quality warning: quantity is unavailable or uncalibrated.")).toBeVisible();
    expect(screen.getAllByText("Missing provenance.")[0]).toHaveAttribute("role", "note");
    expect(screen.getByText("Point estimate has no interval or qualitative uncertainty.")).toBeVisible();
    expect(screen.getByText("Maturity gap: Goal Fixation — dataset backed is false.")).toBeVisible();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
  });

  it("renders unavailable detail quantities without inventing a zero-valued evidence bar", () => {
    const assessment: Assessment = {
      ...assessments[0],
      support: { kind: "confidence" },
    };

    render(<ConceptDetail
      node={ontologyNodes.find((node) => node.id === "failure:goal_fixation")!}
      relations={[]}
      assessments={[assessment]}
      invariants={invariants}
      nodes={ontologyNodes}
    />);

    expect(screen.getByText("Unavailable")).toBeVisible();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^Support:/)).not.toBeInTheDocument();
  });

  it("warns that the shipped Deception Probe and Trace Evidence share a dependency", () => {
    render(<AssessView assessments={assessments} nodes={ontologyNodes} />);

    expect(screen.getByText(/Deception Probe and Trace Evidence share dependency group correlated-trace-and-probe/))
      .toBeVisible();
    expect(screen.getByText(/Correlated evidence is not independent confirmation/)).toBeVisible();
  });

  it("filters invariants and restores the review surface with clear search", async () => {
    const user = userEvent.setup();
    render(<InvariantsView invariants={invariants} />);

    await user.type(screen.getByRole("searchbox", { name: "Search invariants" }), "not-a-governance-key");
    expect(screen.getByText(/No invariants match/)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Clear search" }));

    expect(screen.getByText("The three imperatives are co-equal and mutually constraining.")).toBeVisible();
  });

  it("filters by predicate and traverses a relation without replacing the primary concept", async () => {
    const user = userEvent.setup();
    render(<Workbench />);

    await user.selectOptions(screen.getByLabelText("Relation predicate"), "tests");
    expect(screen.getByText("2 concepts in view")).toBeVisible();

    const relation = screen.getByRole("button", {
      name: "Select relation Goal Flexibility Evaluator tests Goal Fixation",
    });
    relation.focus();
    await user.keyboard("{Enter}");

    expect(screen.getByRole("heading", { name: "Goal Fixation" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Selected relation context" }))
      .toHaveTextContent("Goal Flexibility Evaluator tests Goal Fixation");
  });

  it("keeps export gated for blockers and opens a real invariant review target", async () => {
    const user = userEvent.setup();
    render(<Workbench />);

    await user.click(screen.getByRole("button", { name: "Improve" }));
    expect(screen.getByRole("button", { name: "Generate governed bundle" })).toBeDisabled();
    await user.type(screen.getByLabelText("Relation predicate (optional)"), "achieves");
    await user.type(screen.getByLabelText("Relation target (optional)"), "norm:honesty");
    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));

    expect(screen.getByText("Relation predicate makes a forbidden bridge inference.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Generate governed bundle" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "training-intent-not-success" }));

    const search = screen.getByRole("searchbox", { name: "Search invariants" });
    expect(search).toHaveValue("training-intent-not-success");
    await waitFor(() => expect(search).toHaveFocus());
    expect(screen.getByText("Training intent does not establish training success.")).toBeVisible();
  });

  it("generates a governed bundle with one serialized timestamp and retains checked form state on generation error", async () => {
    const user = userEvent.setup();
    const { unmount } = renderImprove();

    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));
    await user.click(screen.getByRole("button", { name: "Generate governed bundle" }));
    const preview = screen.getByRole("textbox", { name: "Read-only bundle preview" });
    expect((preview as HTMLTextAreaElement).value).toContain('"authority": "generated_noncanonical_bundle"');
    expect((preview as HTMLTextAreaElement).value.match(/"generatedAt": "[^\"]+"/g)).toHaveLength(1);
    expect(screen.getByText("This bundle may inform evaluation or training. It does not establish learned or deployed behavior.")).toBeVisible();
    await user.selectOptions(screen.getByLabelText("Format"), "yaml");
    expect(screen.queryByRole("textbox", { name: "Read-only bundle preview" })).not.toBeInTheDocument();
    expect(screen.queryByText("Governed bundle generated.")).not.toBeInTheDocument();
    unmount();

    renderImprove({ invariants: invariants.slice(0, 21) });
    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));
    await user.click(screen.getByRole("button", { name: "Generate governed bundle" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Could not generate the governed bundle");
    expect(screen.getByLabelText("Label")).toHaveValue("Novel Calibration Evaluator");
    expect(screen.getByText("No blockers or warnings found for this proposal.")).toBeVisible();
  });

  it("disables training export when the selected target lacks curated examples", async () => {
    const user = userEvent.setup();
    renderImprove({ selectedId: "norm:mindfulness" });

    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));
    await user.selectOptions(screen.getByLabelText("Bundle purpose"), "training");

    expect(screen.getByRole("button", { name: "Generate governed bundle" })).toBeDisabled();
    expect(screen.getByText("Training export is unavailable because Mindfulness has no curated behavior examples.")).toBeVisible();
  });

  it("labels validator findings without invariant keys instead of inventing a governance reference", async () => {
    const user = userEvent.setup();
    renderImprove();

    await user.clear(screen.getByLabelText("Canonical ID"));
    await user.type(screen.getByLabelText("Canonical ID"), "failure:goal_fixation");
    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));

    expect(screen.getByText("Proposal ID already exists.")).toBeVisible();
    expect(screen.getByText(/Relevant invariants: No invariant reference/)).toBeVisible();
  });

  it("reports clipboard success and revokes the temporary download URL", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    const createObjectUrl = vi.fn().mockReturnValue("blob:governed-bundle");
    const revokeObjectUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    renderImprove();

    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));
    await user.click(screen.getByRole("button", { name: "Generate governed bundle" }));
    await user.click(screen.getByRole("button", { name: "Copy bundle" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Bundle copied to clipboard."));
    expect(writeText).toHaveBeenCalledWith(expect.stringContaining("generated_noncanonical_bundle"));
    await user.click(screen.getByRole("button", { name: "Download bundle" }));
    await waitFor(() => expect(revokeObjectUrl).toHaveBeenCalledWith("blob:governed-bundle"));
    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
  });

  it("reports clipboard failure without discarding the generated bundle", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: vi.fn().mockRejectedValue(new Error("denied")) } });
    renderImprove();

    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));
    await user.click(screen.getByRole("button", { name: "Generate governed bundle" }));
    await user.click(screen.getByRole("button", { name: "Copy bundle" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Could not copy the bundle."));
    expect(screen.getByRole("textbox", { name: "Read-only bundle preview" })).toBeVisible();
  });

  it("reports download failure, preserves the preview, and revokes a created object URL", async () => {
    const user = userEvent.setup();
    const createObjectUrl = vi.fn().mockReturnValue("blob:failed-governed-bundle");
    const revokeObjectUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectUrl });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {
      throw new Error("download blocked");
    });
    renderImprove();

    await user.click(screen.getByRole("button", { name: "Run semantic checks" }));
    await user.click(screen.getByRole("button", { name: "Generate governed bundle" }));
    const preview = screen.getByRole("textbox", { name: "Read-only bundle preview" });
    const previewValue = (preview as HTMLTextAreaElement).value;
    await user.click(screen.getByRole("button", { name: "Download bundle" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Could not download the bundle.");
    expect(preview).toHaveValue(previewValue);
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:failed-governed-bundle");
  });
});
