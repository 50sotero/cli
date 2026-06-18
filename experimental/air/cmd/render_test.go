package aircmd

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/databricks/cli/libs/cmdio"
	"github.com/databricks/databricks-sdk-go/experimental/mocks"
	"github.com/databricks/databricks-sdk-go/service/jobs"
	"github.com/muesli/termenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// asciiPalette returns a palette whose styles emit no escape codes, so render
// output is plain text and assertions stay readable.
func asciiPalette() palette {
	r := lipgloss.NewRenderer(io.Discard)
	r.SetColorProfile(termenv.Ascii)
	return newPalette(r)
}

func TestConfigYAML(t *testing.T) {
	ctx := t.Context()

	t.Run("inline parameters include the command as a block literal", func(t *testing.T) {
		task := &jobs.GenAiComputeTask{
			YamlParameters: "experiment_name: my-exp\ncompute:\n  accelerator_type: a10\n  num_accelerators: 1\ncommand: \"for i in $(seq 1 3); do echo $i; done\\n\"\n",
		}
		got := configYAML(ctx, mocks.NewMockWorkspaceClient(t).WorkspaceClient, task)
		assert.Contains(t, got, "experiment_name: my-exp")
		assert.Contains(t, got, "accelerator_type: a10")
		assert.Contains(t, got, "command: |")
		assert.Contains(t, got, "  for i in $(seq 1 3); do echo $i; done")
	})

	t.Run("downloads the parameters file when there are no inline parameters", func(t *testing.T) {
		m := mocks.NewMockWorkspaceClient(t)
		m.GetMockWorkspaceAPI().EXPECT().
			Download(mock.Anything, "/Workspace/cfg.yaml").
			Return(io.NopCloser(strings.NewReader("experiment_name: from-file\n")), nil)
		task := &jobs.GenAiComputeTask{YamlParametersFilePath: "/Workspace/cfg.yaml"}
		assert.Equal(t, "experiment_name: from-file", configYAML(ctx, m.WorkspaceClient, task))
	})

	t.Run("falls back to a synthesized config", func(t *testing.T) {
		task := &jobs.GenAiComputeTask{
			MlflowExperimentName: "/Users/me@example.com/exp",
			Compute:              &jobs.ComputeConfig{GpuType: "a10", NumGpus: 1},
		}
		got := configYAML(ctx, mocks.NewMockWorkspaceClient(t).WorkspaceClient, task)
		assert.Contains(t, got, "experiment_name: exp")
		assert.NotContains(t, got, "command")
	})
}

func TestSynthConfigYAML(t *testing.T) {
	task := &jobs.GenAiComputeTask{
		MlflowExperimentName: "/Users/me@example.com/stream-latency-test",
		Compute:              &jobs.ComputeConfig{GpuType: "a10", NumGpus: 1},
	}
	// The accelerator_type uses the raw GPU type; the command is omitted because
	// it lives only in the run parameters.
	want := "experiment_name: stream-latency-test\n" +
		"compute:\n" +
		"  accelerator_type: a10\n" +
		"  num_accelerators: 1"
	assert.Equal(t, want, synthConfigYAML(task))
	assert.Empty(t, synthConfigYAML(&jobs.GenAiComputeTask{}))
}

func TestColorizeConfigLine(t *testing.T) {
	p := asciiPalette()
	// Under the Ascii profile colorization adds no escapes, so each line is
	// preserved verbatim (indentation included) regardless of its role.
	for _, line := range []string{
		"experiment_name: stream-latency-test",
		"compute:",
		"  accelerator_type: a10",
		"  num_accelerators: 1",
		"command: |",
		`  for i in $(seq 1 10); do echo "step $i"; done`,
	} {
		assert.Equal(t, line, colorizeConfigLine(p, line))
	}
}

func TestIsConfigKey(t *testing.T) {
	assert.True(t, isConfigKey("experiment_name"))
	assert.True(t, isConfigKey("num_accelerators"))
	assert.False(t, isConfigKey(""))
	assert.False(t, isConfigKey("for i in $(seq 1 10); do echo "))
	assert.False(t, isConfigKey("Command"))
}

func TestRenderBox(t *testing.T) {
	p := asciiPalette()
	out := renderBox(p, configBoxTitle, "experiment_name: stream-latency-test\ncompute:")
	lines := strings.Split(out, "\n")

	// Title sits in the top border; corners are rounded; every row is the same width.
	assert.Contains(t, lines[0], "╭─ "+configBoxTitle+" ")
	assert.True(t, strings.HasSuffix(lines[0], "╮"))
	assert.True(t, strings.HasPrefix(lines[len(lines)-1], "╰"))
	assert.Contains(t, out, "│  experiment_name: stream-latency-test")

	width := lipgloss.Width(lines[0])
	for _, l := range lines {
		assert.Equal(t, width, lipgloss.Width(l))
	}
}

func TestRenderProgressBar(t *testing.T) {
	p := asciiPalette()
	full := strings.Repeat("█", progressBarWidth)

	t.Run("success is full", func(t *testing.T) {
		got := renderProgressBar(p, "SUCCESS", 10, 10, true, "1m 13s")
		assert.Equal(t, full+"  100%  10/10 steps · 1m 13s", got)
	})

	t.Run("in progress fills floor(width*done/total)", func(t *testing.T) {
		got := renderProgressBar(p, "RUNNING", 5, 10, true, "30s")
		want := strings.Repeat("█", 17) + strings.Repeat("░", 17) + "  50%  5/10 steps · 30s"
		assert.Equal(t, want, got)
	})

	t.Run("no step data drops the percent and steps", func(t *testing.T) {
		got := renderProgressBar(p, "RUNNING", 0, 0, false, "30s")
		assert.Equal(t, strings.Repeat("░", progressBarWidth)+"  30s", got)
	})
}

func TestRenderFields(t *testing.T) {
	p := asciiPalette()
	out := renderFields(p, false, runView{
		runID:        "836121283738861",
		dashboardURL: "https://h.test/jobs/runs/836121283738861",
		status:       "SUCCESS",
		submitted:    "2026-06-03 04:17 UTC",
		retries:      0,
		maxRetries:   "3",
		duration:     "1m 13s",
		experiment:   "stream-latency-test",
		mlflowLabel:  "stream-latency-test",
		mlflowURL:    "https://h.test/ml/experiments/E1/runs/R1",
		user:         "riddhi.bhagwat@databricks.com",
		accelerators: "1x A10",
		environment:  "ml-runtime-gpu:1.0",
	})

	// Labels are padded to the longest ("Accelerators"), so values align.
	assert.Contains(t, out, "Run ID        ")
	assert.Contains(t, out, "Accelerators  1x A10")
	// Max retries and environment show alongside the other fields.
	assert.Contains(t, out, "Max Retries   3")
	assert.Contains(t, out, "Environment   ml-runtime-gpu:1.0")
	// The status carries its dot prefix.
	assert.Contains(t, out, "● SUCCESS")
	// Off a terminal, links render as the bare label (URLs live in JSON output).
	assert.Contains(t, out, "Run ID        836121283738861")
	assert.NotContains(t, out, "https://h.test")
	// The field list is a tight block: no blank lines.
	assert.NotContains(t, out, "\n\n")
}

func TestRenderRunTextProgressVisibility(t *testing.T) {
	ctx := cmdio.MockDiscard(t.Context())
	w := mocks.NewMockWorkspaceClient(t).WorkspaceClient
	render := func(status string) string {
		var buf bytes.Buffer
		// No tasks and no MLflow ids keeps this offline: only the progress box
		// visibility depends on status.
		renderRunText(ctx, &buf, w, &jobs.Run{}, &getData{Status: status, RunID: "1"}, nil)
		return buf.String()
	}

	// A failed run drops the progress box but keeps the metadata.
	failed := render("FAILED")
	assert.NotContains(t, failed, progressBoxTitle)
	assert.Contains(t, failed, metadataBoxTitle)
	assert.Contains(t, failed, "● FAILED")

	// A successful run keeps the progress box.
	assert.Contains(t, render("SUCCESS"), progressBoxTitle)
}

func TestLink(t *testing.T) {
	p := asciiPalette()
	// Color off: the bare label, no URL.
	assert.Equal(t, "label", link(false, p.blue, "label", "https://h.test"))
	assert.Equal(t, "label", link(false, p.blue, "label", ""))
	// With color on, the label is wrapped in an OSC 8 hyperlink to the url.
	assert.Contains(t, link(true, p.blue, "label", "https://h.test"), termenv.Hyperlink("https://h.test", "label"))
}

func TestStatusStyleSelectors(t *testing.T) {
	assert.True(t, isSuccessStatus("SUCCESS"))
	assert.False(t, isSuccessStatus("RUNNING"))
	assert.True(t, isFailedStatus("FAILED"))
	assert.True(t, isFailedStatus("TIMEDOUT"))
	assert.False(t, isFailedStatus("RUNNING"))
}
