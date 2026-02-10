import {
  initializeEditor,
  executeCode,
  setConsoleOutput,
  setCanvas,
  loadExample,
  saveCurrentCode,
  getCodeStore,
} from './editor';

/**
 * Resize canvas to match container dimensions with HighDPI support
 */
function resizeCanvas(canvas: HTMLCanvasElement): void {
  const container = canvas.parentElement;
  if (!container) return;

  const rect = container.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  // Set canvas buffer size to match display size * device pixel ratio
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);

  // Note: CSS handles the display size, we just set the buffer size
  console.log(`Canvas resized to: ${canvas.width}x${canvas.height} (DPR: ${dpr})`);
}

async function main(): Promise<void> {
  try {
    const errorElement = document.getElementById('error-message') as HTMLDivElement;
    const consoleOutput = document.getElementById('console-output') as HTMLDivElement;
    const compileBtn = document.getElementById('compile-btn') as HTMLButtonElement;
    const exampleSelect = document.getElementById('example-select') as HTMLSelectElement;
    const canvas = document.getElementById('webgpu-canvas') as HTMLCanvasElement;

    if (!errorElement || !consoleOutput || !compileBtn || !exampleSelect || !canvas) {
      throw new Error('Required DOM elements not found');
    }

    // Set up console output handler
    setConsoleOutput(consoleOutput);
    setCanvas(canvas);

    // Initialize editor
    const editor = await initializeEditor();

    // Resize canvas initially
    resizeCanvas(canvas);

    // Set up ResizeObserver to handle container resizing
    const resizeObserver = new ResizeObserver(() => {
      resizeCanvas(canvas);
      // Re-execute code to reinitialize WebGPU with new dimensions
      const code = editor.getValue();
      executeCode(code);
    });

    const canvasContainer = canvas.parentElement;
    if (canvasContainer) {
      resizeObserver.observe(canvasContainer);
    }

    // Set up compile button click handler
    compileBtn.addEventListener('click', async () => {
      const code = editor.getValue();
      saveCurrentCode(code);
      await executeCode(code);
    });

    // Set up example dropdown change handler
    exampleSelect.addEventListener('change', async () => {
      const selectedExample = exampleSelect.value;
      saveCurrentCode(editor.getValue());
      loadExample(selectedExample);
      const codeStore = getCodeStore();
      await executeCode(codeStore[selectedExample]);
    });

    // Execute initial code on load
    const initialCode = editor.getValue();
    await executeCode(initialCode);
  } catch (error) {
    console.error('Failed to initialize:', error);
    const errorElement = document.getElementById('error-message') as HTMLDivElement;
    if (errorElement) {
      errorElement.textContent = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
      errorElement.classList.remove('hidden');
    }
  }
}

main().catch(console.error);
