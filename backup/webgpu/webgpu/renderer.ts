import type { GPUContext } from '@/types';
import { triangleShaders, createShaderModule } from './shaders';

export class Renderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private pipeline: GPURenderPipeline | null = null;
  private vertexBuffer: GPUBuffer | null = null;

  constructor(gpuContext: GPUContext) {
    this.device = gpuContext.device;
    this.context = gpuContext.context;
    this.format = gpuContext.format;
  }

  async initialize(): Promise<void> {
    const vertexShader = await createShaderModule(this.device, triangleShaders.vertex);
    const fragmentShader = await createShaderModule(this.device, triangleShaders.fragment);

    const vertices = new Float32Array([
      // Position, Color
      0.0,
      0.5,
      0.0,
      1.0,
      0.0,
      0.0, // Top - Red
      -0.5,
      -0.5,
      0.0,
      0.0,
      1.0,
      0.0, // Bottom Left - Green
      0.5,
      -0.5,
      0.0,
      0.0,
      0.0,
      1.0, // Bottom Right - Blue
    ]);

    this.vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);

    const vertexBufferLayout: GPUVertexBufferLayout = {
      arrayStride: 24, // 6 floats * 4 bytes
      attributes: [
        {
          format: 'float32x3',
          offset: 0,
          shaderLocation: 0,
        },
        {
          format: 'float32x3',
          offset: 12,
          shaderLocation: 1,
        },
      ],
    };

    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: vertexShader,
        entryPoint: 'main',
        buffers: [vertexBufferLayout],
      },
      fragment: {
        module: fragmentShader,
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
    if (!this.pipeline || !this.vertexBuffer) {
      // Silently skip rendering if not initialized - this allows the render loop to continue
      // while waiting for valid user code
      return;
    }

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPass: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPass);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.draw(3, 1, 0, 0);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}
