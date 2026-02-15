// WebGPU Rotating Cube
// This code creates a rotating 3D cube with depth testing
// Uses wgpu-matrix library for matrix math

// Vertex shader with uniforms
const vertexShaderCode = `
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
`;

// Fragment shader
const fragmentShaderCode = `
  struct VS_to_FS {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
  };

  @fragment
  fn main(input: VS_to_FS) -> @location(0) vec4<f32> {
    return input.color;
  }
`;

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
      -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0,
      -0.5, 0.5, 0.5, 1.0, 0.0, 0.0,
      // Back face (green)
      -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.5, 0.5, -0.5, 0.0, 1.0,
      0.0, 0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
      // Top face (blue)
      -0.5, 0.5, -0.5, 0.0, 0.0, 1.0, -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
      0.5, 0.5, -0.5, 0.0, 0.0, 1.0,
      // Bottom face (yellow)
      -0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 0.5, -0.5, 0.5, 1.0, 1.0,
      0.0, -0.5, -0.5, 0.5, 1.0, 1.0, 0.0,
      // Right face (magenta)
      0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 0.5, 0.5, -0.5, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
      0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
      // Left face (cyan)
      -0.5, -0.5, -0.5, 0.0, 1.0, 1.0, -0.5, -0.5, 0.5, 0.0, 1.0, 1.0, -0.5, 0.5, 0.5, 0.0, 1.0,
      1.0, -0.5, 0.5, -0.5, 0.0, 1.0, 1.0,
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
      0,
      1,
      2,
      0,
      2,
      3, // Front
      4,
      5,
      6,
      4,
      6,
      7, // Back
      8,
      9,
      10,
      8,
      10,
      11, // Top
      12,
      13,
      14,
      12,
      14,
      15, // Bottom
      16,
      17,
      18,
      16,
      18,
      19, // Right
      20,
      21,
      22,
      20,
      22,
      23, // Left
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
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      mvp.buffer,
      mvp.byteOffset,
      mvp.byteLength
    );
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
