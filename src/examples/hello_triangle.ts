// WebGPU Hello Triangle
// This code creates a colored triangle using WebGPU

// Shared shader code
const sharedShaderCode = `
  struct VS_to_FS {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
  };
`;

// Vertex shader - positions and colors
const vertexShaderCode =
  sharedShaderCode +
  `
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
`;

// Fragment shader - outputs color
const fragmentShaderCode =
  sharedShaderCode +
  `
  @fragment
  fn main(input : VS_to_FS) -> @location(0) vec4<f32> {
    return input.color; // Red color
  }
`;

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
