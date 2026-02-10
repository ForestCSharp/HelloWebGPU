import * as monaco from 'monaco-editor';
import loader from '@monaco-editor/loader';
import type { editor } from 'monaco-editor';
import * as ts from 'typescript';
import webgpuTypes from '../node_modules/@webgpu/types/dist/index.d.ts?raw';

let editorInstance: editor.IStandaloneCodeEditor | null = null;
let consoleElement: HTMLDivElement | null = null;
let canvasElement: HTMLCanvasElement | null = null;
let animationId: number | null = null;
let currentExampleId: string = 'example1';

// Store for editable code examples
const codeStore: Record<string, string> = {
  example1: getInitialCode(),
  example2: getInitialCode(),
};

export function getCodeStore(): Record<string, string> {
  return codeStore;
}

export function getCurrentExampleId(): string {
  return currentExampleId;
}

export function setCurrentExampleId(id: string): void {
  currentExampleId = id;
}

export function saveCurrentCode(code: string): void {
  codeStore[currentExampleId] = code;
}

export function loadExample(exampleId: string): void {
  if (!codeStore[exampleId]) {
    console.error(`Example ${exampleId} not found`);
    return;
  }

  currentExampleId = exampleId;

  if (editorInstance) {
    editorInstance.setValue(codeStore[exampleId]);
  }
}

export function setConsoleOutput(element: HTMLDivElement): void {
  consoleElement = element;
}

export function setCanvas(element: HTMLCanvasElement): void {
  canvasElement = element;
}

function logToConsole(message: string, type: 'log' | 'error' | 'warn' = 'log'): void {
  if (!consoleElement) return;

  const line = document.createElement('div');
  line.className = `console-line console-${type}`;
  line.textContent = message;
  consoleElement.appendChild(line);
  consoleElement.scrollTop = consoleElement.scrollHeight;
}

function clearConsole(): void {
  if (!consoleElement) return;
  consoleElement.innerHTML = '';
}

export async function initializeEditor(): Promise<editor.IStandaloneCodeEditor> {
  loader.config({ monaco });

  await loader.init();

  // Configure TypeScript compiler options
  const tsLang = monaco.languages.typescript as any;
  tsLang.typescriptDefaults.setCompilerOptions({
    target: tsLang.ScriptTarget.ES2022,
    module: tsLang.ModuleKind.CommonJS,
    lib: ['es2022', 'dom'],
    strict: true,
    esModuleInterop: true,
    skipLibCheck: true,
  });

  // Add WebGPU types as extra lib
  tsLang.typescriptDefaults.addExtraLib(webgpuTypes, 'file:///webgpu-types.d.ts');

  // Create editor
  const editorContainer = document.getElementById('monaco-editor');
  if (!editorContainer) {
    throw new Error('Monaco editor container not found');
  }

  const editor = monaco.editor.create(editorContainer, {
    value: codeStore[currentExampleId],
    language: 'typescript',
    theme: 'vs-dark',
    automaticLayout: true,
    minimap: { enabled: false },
    fontSize: 14,
    lineNumbers: 'on',
    scrollBeyondLastLine: false,
    wordWrap: 'on',
  });

  editorInstance = editor;
  return editor;
}

export function getEditorInstance(): editor.IStandaloneCodeEditor | null {
  return editorInstance;
}

/**
 * Transpile TypeScript code to JavaScript using TypeScript compiler
 */
function transpileTypeScript(tsCode: string): {
  jsCode: string;
  errors: Array<{ line: number; message: string }>;
} {
  const result = ts.transpileModule(tsCode, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.Preserve,
      isolatedModules: true,
      esModuleInterop: true,
    },
    reportDiagnostics: true,
  });

  const errors: Array<{ line: number; message: string }> = [];

  if (result.diagnostics) {
    for (const diagnostic of result.diagnostics) {
      const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
      const lineMatch = message.match(/\((\d+),\d+\)/);
      const line = lineMatch ? parseInt(lineMatch[1]) : 1;
      errors.push({ line, message });
    }
  }

  return { jsCode: result.outputText, errors };
}

export async function executeCode(code: string): Promise<void> {
  try {
    clearConsole();

    // Stop any existing animation
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }

    // Transpile TypeScript to JavaScript
    const { jsCode, errors } = transpileTypeScript(code);

    // If there are compilation errors, show the first one
    if (errors.length > 0) {
      const firstError = errors[0];
      throw new Error(`Line ${firstError.line}: ${firstError.message}`);
    }

    // Get canvas and set up WebGPU
    if (!canvasElement) {
      throw new Error('Canvas not found');
    }

    // Initialize WebGPU
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('Failed to request WebGPU adapter');
    }

    const device = await adapter.requestDevice();
    const context = canvasElement.getContext('webgpu');
    if (!context) {
      throw new Error('Failed to get WebGPU context');
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
    });

    // Capture console output
    const originalLog = console.log;
    const originalError = console.error;
    const originalWarn = console.warn;

    console.log = (...args) => {
      logToConsole(args.map(a => String(a)).join(' '), 'log');
      originalLog.apply(console, args);
    };

    console.error = (...args) => {
      logToConsole(args.map(a => String(a)).join(' '), 'error');
      originalError.apply(console, args);
    };

    console.warn = (...args) => {
      logToConsole(args.map(a => String(a)).join(' '), 'warn');
      originalWarn.apply(console, args);
    };

    try {
      // Strip imports and exports for execution in Function context
      const executableCode = jsCode
        .replace(/import\s+.*?\n+from\s+['"].*?['"];?\n*/g, '')
        .replace(/export\s+/g, '');

      // Execute the code with WebGPU context available
      const runCode = new Function('device', 'context', 'format', executableCode);
      const renderer = runCode(device, context, format);

      // Set up render loop if the code returns a renderer with render method
      if (renderer && typeof renderer.render === 'function') {
        function renderLoop(): void {
          try {
            renderer.render();
            animationId = requestAnimationFrame(renderLoop);
          } catch (e) {
            console.error('Render error:', e);
          }
        }
        renderLoop();
      }

      // Hide any previous error messages
      const errorElement = document.getElementById('error-message') as HTMLDivElement;
      if (errorElement) {
        errorElement.classList.add('hidden');
        errorElement.textContent = '';
      }

      console.log('WebGPU initialized successfully');
    } finally {
      // Restore console methods
      console.log = originalLog;
      console.error = originalError;
      console.warn = originalWarn;
    }
  } catch (error) {
    console.error('Failed to execute code:', error);

    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logToConsole(`Error: ${errorMessage}`, 'error');

    const errorElement = document.getElementById('error-message') as HTMLDivElement;
    if (errorElement) {
      let displayMessage = errorMessage;
      let lineNumber: string | null = null;

      if (errorMessage.includes('Line ')) {
        const lineMatch = errorMessage.match(/Line (\d+):/);
        if (lineMatch) {
          lineNumber = lineMatch[1];
          const errorWithoutLine = errorMessage.replace(/Line \d+:/, '').trim();
          displayMessage = `<span class="error-line-number">Line ${lineNumber}</span><span class="error-message">${errorWithoutLine}</span>`;

          const editor = getEditorInstance();
          if (editor) {
            const lineIndex = parseInt(lineNumber);
            editor.setPosition({ lineNumber: lineIndex, column: 1 });
            editor.revealLineInCenter(lineIndex);
            editor.focus();
          }
        }
      } else {
        displayMessage = `<span class="error-message">Error: ${errorMessage}</span>`;
      }

      errorElement.innerHTML = displayMessage;
      errorElement.classList.remove('hidden');
    }
  }
}

function getInitialCode(): string {
  return `// WebGPU Hello Triangle
// This code creates a colored triangle using WebGPU

// Shared shader code
const sharedShaderCode = \`
  struct VS_to_FS {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
  };
\`;

// Vertex shader - positions and colors
const vertexShaderCode = 
sharedShaderCode 
+
\`
  @vertex
  fn main(@builtin(vertex_index) vertexIndex: u32) -> VS_to_FS
  {
    var positions = array<vec2<f32>, 3>(
      vec2<f32>(0.0, 0.5),
      vec2<f32>(-0.5, -0.5),
      vec2<f32>(0.5, -0.5)
    );

    var colors = array<vec3<f32>, 3>(
      vec3<f32>(1.0, 0.0, 0.0),
      vec3<f32>(0.0, 1.0, 0.0),
      vec3<f32>(0.0, 0.0, 1.0)
    );

    var output : VS_to_FS;
    output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
    output.color = vec4<f32>(colors[vertexIndex], 1.0);
    return output;
  }
\`;

// Fragment shader - outputs color
const fragmentShaderCode = 
sharedShaderCode 
+
\`
  @fragment
  fn main(input : VS_to_FS) -> @location(0) vec4<f32> {
    return input.color; // Red color
  }
\`;

class Renderer {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  pipeline: GPURenderPipeline;

  constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat) {
    this.device = device;
    this.context = context;
    this.format = format;
    
    this.initialize();
  }

  initialize(): void {
    // Create shader modules
    const vertexShaderModule = this.device.createShaderModule({
      code: vertexShaderCode,
    });

    const fragmentShaderModule = this.device.createShaderModule({
      code: fragmentShaderCode,
    });

    // Create render pipeline
    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: vertexShaderModule,
        entryPoint: 'main',
      },
      fragment: {
        module: fragmentShaderModule,
        entryPoint: 'main',
        targets: [
          {
            format: this.format,
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }

  render(): void {
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.draw(3);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}

// Create and return renderer
const renderer = new Renderer(device, context, format);
console.log('Renderer created successfully');

return renderer;
`;
}
