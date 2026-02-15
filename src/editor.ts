import * as monaco from 'monaco-editor';
import loader from '@monaco-editor/loader';
import type { editor } from 'monaco-editor';
import * as ts from 'typescript';
import * as wgpuMatrix from 'wgpu-matrix';
import webgpuTypes from '../node_modules/@webgpu/types/dist/index.d.ts?raw';

let editorInstance: editor.IStandaloneCodeEditor | null = null;
let consoleElement: HTMLDivElement | null = null;
let canvasElement: HTMLCanvasElement | null = null;
let animationId: number | null = null;
let currentExampleId: string = 'hello_triangle';

// Store for editable code examples
const codeStore: Record<string, string> = {
  hello_triangle: getInitialCode(),
  rotating_cube: getCubeCode(),
  gltf_model: getGltfCode(),
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
    fontSize: 12,
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
      const runCode = new Function('device', 'context', 'format', 'mat4', executableCode);
      const renderer = runCode(device, context, format, wgpuMatrix.mat4);

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

function getCubeCode(): string {
  return `// WebGPU Rotating Cube
// This code creates a rotating 3D cube with depth testing
// Uses wgpu-matrix library for matrix math

// Vertex shader with uniforms
const vertexShaderCode = \`
  struct Uniforms {
    mvpMatrix: mat4x4<f32>,
  };

  @binding(0) @group(0) var<uniform> uniforms: Uniforms;

  struct VS_to_FS {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
  };

  @vertex
  fn main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>
  ) -> VS_to_FS {
    var output: VS_to_FS;
    output.position = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
    output.color = vec4<f32>(color, 1.0);
    return output;
  }
\`;

// Fragment shader
const fragmentShaderCode = \`
  struct VS_to_FS {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
  };

  @fragment
  fn main(input: VS_to_FS) -> @location(0) vec4<f32> {
    return input.color;
  }
\`;

class Renderer {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  pipeline: GPURenderPipeline;
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  depthTexture: GPUTexture;
  canvasWidth: number;
  canvasHeight: number;
  startTime: number;

  constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.canvasWidth = 800;
    this.canvasHeight = 600;
    this.startTime = performance.now();
    
    this.initialize();
  }

  initialize(): void {
    // Cube vertices: position (3 floats) + color (3 floats)
    // Each face has different color
    const vertices = new Float32Array([
      // Front face (red)
      -0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
       0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
       0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
      -0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
      // Back face (green)
      -0.5, -0.5, -0.5,   0.0, 1.0, 0.0,
      -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
       0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
       0.5, -0.5, -0.5,   0.0, 1.0, 0.0,
      // Top face (blue)
      -0.5,  0.5, -0.5,   0.0, 0.0, 1.0,
      -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
       0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
       0.5,  0.5, -0.5,   0.0, 0.0, 1.0,
      // Bottom face (yellow)
      -0.5, -0.5, -0.5,   1.0, 1.0, 0.0,
       0.5, -0.5, -0.5,   1.0, 1.0, 0.0,
       0.5, -0.5,  0.5,   1.0, 1.0, 0.0,
      -0.5, -0.5,  0.5,   1.0, 1.0, 0.0,
      // Right face (magenta)
       0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
       0.5,  0.5, -0.5,   1.0, 0.0, 1.0,
       0.5,  0.5,  0.5,   1.0, 0.0, 1.0,
       0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
      // Left face (cyan)
      -0.5, -0.5, -0.5,   0.0, 1.0, 1.0,
      -0.5, -0.5,  0.5,   0.0, 1.0, 1.0,
      -0.5,  0.5,  0.5,   0.0, 1.0, 1.0,
      -0.5,  0.5, -0.5,   0.0, 1.0, 1.0,
    ]);

    // Create vertex buffer
    this.vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    // Cube indices (6 faces * 2 triangles * 3 vertices = 36 indices)
    const indices = new Uint16Array([
       0,  1,  2,   0,  2,  3,  // Front
       4,  5,  6,   4,  6,  7,  // Back
       8,  9, 10,   8, 10, 11,  // Top
      12, 13, 14,  12, 14, 15,  // Bottom
      16, 17, 18,  16, 18, 19,  // Right
      20, 21, 22,  20, 22, 23,  // Left
    ]);

    // Create index buffer
    this.indexBuffer = this.device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    // Create uniform buffer (mat4x4<f32> = 16 floats * 4 bytes = 64 bytes)
    this.uniformBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create depth texture
    this.createDepthTexture();

    // Create shader modules
    const vertexShaderModule = this.device.createShaderModule({
      code: vertexShaderCode,
    });

    const fragmentShaderModule = this.device.createShaderModule({
      code: fragmentShaderCode,
    });

    // Create bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform' },
        },
      ],
    });

    // Create bind group
    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.uniformBuffer },
        },
      ],
    });

    // Create render pipeline
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: vertexShaderModule,
        entryPoint: 'main',
        buffers: [
          {
            arrayStride: 24, // 6 floats * 4 bytes
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
              },
              {
                shaderLocation: 1,
                offset: 12, // 3 floats * 4 bytes
                format: 'float32x3',
              },
            ],
          },
        ],
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
        cullMode: 'back',
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
    });
  }

  createDepthTexture(): void {
    this.depthTexture = this.device.createTexture({
      size: { width: this.canvasWidth, height: this.canvasHeight },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  updateUniforms(): void {
    const currentTime = performance.now();
    const elapsed = (currentTime - this.startTime) / 1000;

    // Rotation angles
    const angleY = elapsed * 1.0; // 1 radian per second around Y
    const angleX = elapsed * 0.5; // 0.5 radian per second around X

    // Create matrices using wgpu-matrix
    const model = mat4.multiply(mat4.rotationY(angleY), mat4.rotationX(angleX));
    const view = mat4.translation([0, 0, -2.5]);
    const aspect = this.canvasWidth / this.canvasHeight;
    const projection = mat4.perspective(Math.PI / 4, aspect, 0.1, 100.0);

    // Combine: MVP = P * V * M
    const mvp = mat4.multiply(mat4.multiply(projection, view), model);

    // Update uniform buffer (wgpu-matrix returns column-major, which is what WGSL expects)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, mvp.buffer, mvp.byteOffset, mvp.byteLength);
  }

  render(): void {
    // Check if canvas size changed and recreate depth texture if needed
    const canvas = this.context.canvas as HTMLCanvasElement;
    if (canvas.width !== this.canvasWidth || canvas.height !== this.canvasHeight) {
      this.canvasWidth = canvas.width;
      this.canvasHeight = canvas.height;
      this.depthTexture.destroy();
      this.createDepthTexture();
    }

    this.updateUniforms();

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint16');
    passEncoder.drawIndexed(36);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}

// Create and return renderer
const renderer = new Renderer(device, context, format);
console.log('Rotating cube renderer created successfully');

return renderer;
`;
}

function getGltfCode(): string {
  return `// WebGPU glTF Model Renderer
// This code loads and renders the Damaged Helmet glTF model
// Uses wgpu-matrix library for matrix math

// Vertex shader with uniforms
const vertexShaderCode = \`
  struct Uniforms {
    mvpMatrix: mat4x4<f32>,
    normalMatrix: mat4x4<f32>,
  };

  @binding(0) @group(0) var<uniform> uniforms: Uniforms;

  struct VS_to_FS {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
  };

  @vertex
  fn main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>
  ) -> VS_to_FS {
    var output: VS_to_FS;
    output.position = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
    output.normal = (uniforms.normalMatrix * vec4<f32>(normal, 0.0)).xyz;
    output.uv = uv;
    return output;
  }
\`;

// Fragment shader with simple lighting
const fragmentShaderCode = \`
  struct VS_to_FS {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
  };

  @fragment
  fn main(input: VS_to_FS) -> @location(0) vec4<f32> {
    // Simple directional lighting
    let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let normal = normalize(input.normal);
    let diff = max(dot(normal, lightDir), 0.0);
    let ambient = 0.3;
    let lighting = ambient + diff * 0.7;
    
    // Checkerboard pattern based on UV
    let checker = (floor(input.uv.x * 10.0) + floor(input.uv.y * 10.0)) % 2.0;
    let baseColor = mix(vec3<f32>(0.8, 0.8, 0.8), vec3<f32>(0.5, 0.5, 0.5), checker);
    
    return vec4<f32>(baseColor * lighting, 1.0);
  }
\`;

// glTF types
interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  max?: number[];
  min?: number[];
}

interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number;
}

interface GltfBuffer {
  uri?: string;
  byteLength: number;
}

interface GltfMeshPrimitive {
  attributes: Record<string, number>;
  indices?: number;
  mode?: number;
}

interface GltfMesh {
  primitives: GltfMeshPrimitive[];
}

interface GltfNode {
  mesh?: number;
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
  children?: number[];
}

interface GltfScene {
  nodes: number[];
}

interface GltfData {
  asset: { version: string };
  scene?: number;
  scenes: GltfScene[];
  nodes: GltfNode[];
  meshes: GltfMesh[];
  accessors: GltfAccessor[];
  bufferViews: GltfBufferView[];
  buffers: GltfBuffer[];
}

// Synchronous fetch function using XMLHttpRequest
function fetchSync(url: string): string {
  const xhr = new XMLHttpRequest();
  xhr.open('GET', url, false); // false = synchronous
  xhr.send();
  if (xhr.status !== 200) {
    throw new Error('Failed to load ' + url + ': ' + xhr.status);
  }
  return xhr.responseText;
}

function fetchArrayBufferSync(url: string): ArrayBuffer {
  const xhr = new XMLHttpRequest();
  xhr.open('GET', url, false);
  // Trick: overrideMimeType must be called before send() for sync XHR to get binary data as text
  xhr.overrideMimeType('text/plain; charset=x-user-defined');
  xhr.send();
  if (xhr.status !== 200) {
    throw new Error('Failed to load ' + url + ': ' + xhr.status);
  }
  // Convert text response to ArrayBuffer
  const text = xhr.responseText;
  const buffer = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) {
    buffer[i] = text.charCodeAt(i) & 0xff;
  }
  return buffer.buffer;
}

class GltfLoader {
  load(url: string): GltfData {
    const responseText = fetchSync(url);
    const gltf: GltfData = JSON.parse(responseText);
    return gltf;
  }
}

class GltfRenderer {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  pipeline!: GPURenderPipeline;
  uniformBuffer!: GPUBuffer;
  bindGroup!: GPUBindGroup;
  depthTexture!: GPUTexture;
  canvasWidth: number;
  canvasHeight: number;
  startTime: number;
  
  // Mesh data
  meshes: Array<{
    vertexBuffer: GPUBuffer;
    indexBuffer?: GPUBuffer;
    indexCount: number;
    hasIndices: boolean;
  }> = [];

  constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.canvasWidth = 800;
    this.canvasHeight = 600;
    this.startTime = performance.now();
    
    this.initialize();
  }

  initialize(): void {
    // Create depth texture
    this.createDepthTexture();

    // Create uniform buffer (mat4x4<f32> * 2 = 32 floats * 4 bytes = 128 bytes)
    this.uniformBuffer = this.device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create shader modules
    const vertexShaderModule = this.device.createShaderModule({
      code: vertexShaderCode,
    });

    const fragmentShaderModule = this.device.createShaderModule({
      code: fragmentShaderCode,
    });

    // Create bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform' },
        },
      ],
    });

    // Create bind group
    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.uniformBuffer },
        },
      ],
    });

    // Create render pipeline
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: vertexShaderModule,
        entryPoint: 'main',
        buffers: [
          {
            arrayStride: 32, // 8 floats * 4 bytes (position + normal + uv)
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
              },
              {
                shaderLocation: 1,
                offset: 12, // 3 floats * 4 bytes
                format: 'float32x3',
              },
              {
                shaderLocation: 2,
                offset: 24, // 6 floats * 4 bytes
                format: 'float32x2',
              },
            ],
          },
        ],
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
        cullMode: 'back',
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
    });

    // Load glTF model
    this.loadGltfModel();
  }

  loadGltfModel(): void {
    console.log('Loading Damaged Helmet glTF model...');
    
    // Load the glTF JSON
    const loader = new GltfLoader();
    const gltf = loader.load('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf');
    
    console.log('glTF loaded:', gltf);
    
    // Load all buffers
    const buffers: ArrayBuffer[] = [];
    if (gltf.buffers && Array.isArray(gltf.buffers)) {
      for (let i = 0; i < gltf.buffers.length; i++) {
        const buffer = gltf.buffers[i];
        if (buffer && buffer.uri) {
          const baseUrl = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/';
          const bufferUrl = baseUrl + buffer.uri;
          const data = fetchArrayBufferSync(bufferUrl);
          buffers.push(data);
          console.log('Buffer ' + i + ' loaded: ' + data.byteLength + ' bytes');
        }
      }
    }

    // Process meshes
    if (gltf.nodes && Array.isArray(gltf.nodes)) {
      for (const node of gltf.nodes) {
        if (node && node.mesh !== undefined && gltf.meshes && Array.isArray(gltf.meshes)) {
          const mesh = gltf.meshes[node.mesh];
          if (mesh && mesh.primitives && Array.isArray(mesh.primitives)) {
            for (const primitive of mesh.primitives) {
              this.processPrimitive(gltf, primitive, buffers);
            }
          }
        }
      }
    }
    
    console.log('Loaded ' + this.meshes.length + ' mesh primitives');
  }

  processPrimitive(
    gltf: GltfData, 
    primitive: GltfMeshPrimitive, 
    buffers: ArrayBuffer[]
  ): void {
    // Get accessor indices
    const positionAccessorIdx = primitive.attributes?.POSITION;
    const normalAccessorIdx = primitive.attributes?.NORMAL;
    const texcoordAccessorIdx = primitive.attributes?.TEXCOORD_0;
    const indicesAccessorIdx = primitive.indices;

    if (positionAccessorIdx === undefined) {
      console.warn('Primitive missing POSITION attribute');
      return;
    }

    if (!gltf.accessors || !Array.isArray(gltf.accessors)) {
      console.warn('glTF missing accessors');
      return;
    }

    const positionAccessor = gltf.accessors[positionAccessorIdx];
    const normalAccessor = normalAccessorIdx !== undefined ? gltf.accessors[normalAccessorIdx] : null;
    const texcoordAccessor = texcoordAccessorIdx !== undefined ? gltf.accessors[texcoordAccessorIdx] : null;

    if (!positionAccessor) {
      console.warn('Position accessor not found');
      return;
    }

    // Extract vertex data
    const positions = this.readAccessor(gltf, positionAccessor, buffers, 3);
    const normals = normalAccessor ? this.readAccessor(gltf, normalAccessor, buffers, 3) : null;
    const uvs = texcoordAccessor ? this.readAccessor(gltf, texcoordAccessor, buffers, 2) : null;

    const vertexCount = positionAccessor.count;
    
    // Build interleaved vertex buffer
    const vertexData = new Float32Array(vertexCount * 8); // pos(3) + normal(3) + uv(2)
    
    for (let i = 0; i < vertexCount; i++) {
      // Position
      vertexData[i * 8 + 0] = positions[i * 3 + 0];
      vertexData[i * 8 + 1] = positions[i * 3 + 1];
      vertexData[i * 8 + 2] = positions[i * 3 + 2];
      
      // Normal
      if (normals) {
        vertexData[i * 8 + 3] = normals[i * 3 + 0];
        vertexData[i * 8 + 4] = normals[i * 3 + 1];
        vertexData[i * 8 + 5] = normals[i * 3 + 2];
      } else {
        vertexData[i * 8 + 3] = 0;
        vertexData[i * 8 + 4] = 1;
        vertexData[i * 8 + 5] = 0;
      }
      
      // UV
      if (uvs) {
        vertexData[i * 8 + 6] = uvs[i * 2 + 0];
        vertexData[i * 8 + 7] = uvs[i * 2 + 1];
      } else {
        vertexData[i * 8 + 6] = 0;
        vertexData[i * 8 + 7] = 0;
      }
    }

    // Create vertex buffer
    const vertexBuffer = this.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertexData);
    vertexBuffer.unmap();

    let indexBuffer: GPUBuffer | undefined;
    let indexCount = vertexCount;

    // Create index buffer if available
    if (indicesAccessorIdx !== undefined && gltf.accessors && Array.isArray(gltf.accessors)) {
      const indicesAccessor = gltf.accessors[indicesAccessorIdx];
      if (indicesAccessor) {
        const indices = this.readAccessor(gltf, indicesAccessor, buffers, 1);
        
        // Convert to Uint16Array if needed
        let indexData: Uint16Array;
        if (indices instanceof Uint16Array) {
          indexData = indices;
        } else {
          indexData = new Uint16Array(indices.length);
          for (let i = 0; i < indices.length; i++) {
            indexData[i] = indices[i];
          }
        }
        
        indexBuffer = this.device.createBuffer({
          size: indexData.byteLength,
          usage: GPUBufferUsage.INDEX,
          mappedAtCreation: true,
        });
        new Uint16Array(indexBuffer.getMappedRange()).set(indexData);
        indexBuffer.unmap();
        
        indexCount = indicesAccessor.count;
      }
    }

    this.meshes.push({
      vertexBuffer,
      indexBuffer,
      indexCount,
      hasIndices: indexBuffer !== undefined,
    });
  }

  readAccessor(
    gltf: GltfData,
    accessor: GltfAccessor,
    buffers: ArrayBuffer[],
    components: number
  ): Float32Array | Uint16Array | Uint32Array {
    if (!gltf.bufferViews || !Array.isArray(gltf.bufferViews)) {
      throw new Error('glTF missing bufferViews');
    }
    
    const bufferView = gltf.bufferViews[accessor.bufferView || 0];
    if (!bufferView) {
      throw new Error('BufferView not found: ' + (accessor.bufferView || 0));
    }
    
    const buffer = buffers[bufferView.buffer];
    if (!buffer) {
      throw new Error('Buffer not found: ' + bufferView.buffer);
    }
    
    const byteOffset = (accessor.byteOffset || 0) + (bufferView.byteOffset || 0);
    const count = accessor.count;
    
    // Map component type to TypedArray
    switch (accessor.componentType) {
      case 5126: // FLOAT
        return new Float32Array(buffer, byteOffset, count * components);
      case 5123: // UNSIGNED_SHORT
        return new Uint16Array(buffer, byteOffset, count * components);
      case 5125: // UNSIGNED_INT
        return new Uint32Array(buffer, byteOffset, count * components);
      default:
        console.warn('Unknown component type:', accessor.componentType);
        return new Float32Array(buffer, byteOffset, count * components);
    }
  }

  createDepthTexture(): void {
    this.depthTexture = this.device.createTexture({
      size: { width: this.canvasWidth, height: this.canvasHeight },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  updateUniforms(): void {
    const currentTime = performance.now();
    const elapsed = (currentTime - this.startTime) / 1000;

    // Rotation around Y axis (turntable style)
    const angleY = elapsed * 0.5;

    // Create matrices using wgpu-matrix
    // First rotate 90 deg around X to make model face forward (instead of up)
    // Then rotate around Y for the turntable animation
    const correction = mat4.rotationX(Math.PI / 2);
    const rotation = mat4.rotationY(angleY);
    const model = mat4.multiply(rotation, correction);
    const view = mat4.translation([0, 0, -3.5]);
    const aspect = this.canvasWidth / this.canvasHeight;
    const projection = mat4.perspective(Math.PI / 4, aspect, 0.1, 100.0);

    // Combine: MVP = P * V * M
    const mvp = mat4.multiply(mat4.multiply(projection, view), model);
    
    // Normal matrix (inverse transpose of model)
    const normalMatrix = mat4.multiply(mat4.multiply(projection, view), mat4.identity());

    // Update uniform buffer
    this.device.queue.writeBuffer(this.uniformBuffer, 0, mvp.buffer, mvp.byteOffset, mvp.byteLength);
    this.device.queue.writeBuffer(this.uniformBuffer, 64, normalMatrix.buffer, normalMatrix.byteOffset, normalMatrix.byteLength);
  }

  render(): void {
    // Check if canvas size changed and recreate depth texture if needed
    const canvas = this.context.canvas as HTMLCanvasElement;
    if (canvas.width !== this.canvasWidth || canvas.height !== this.canvasHeight) {
      this.canvasWidth = canvas.width;
      this.canvasHeight = canvas.height;
      this.depthTexture.destroy();
      this.createDepthTexture();
    }

    this.updateUniforms();

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);

    // Render all meshes
    for (const mesh of this.meshes) {
      passEncoder.setVertexBuffer(0, mesh.vertexBuffer);
      if (mesh.hasIndices && mesh.indexBuffer) {
        passEncoder.setIndexBuffer(mesh.indexBuffer, 'uint16');
        passEncoder.drawIndexed(mesh.indexCount);
      } else {
        passEncoder.draw(mesh.indexCount);
      }
    }

    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }
}

// Create and return renderer
const renderer = new GltfRenderer(device, context, format);
console.log('glTF renderer created successfully');

return renderer;
`;
}
