use bytemuck;
use wgpu::util::DeviceExt;

const N: usize = 1024;
const WORKGROUP_SIZE: usize = 8;
const BUFF_SIZE: wgpu::BufferAddress = (std::mem::size_of::<f32>() * N * N) as wgpu::BufferAddress;

#[derive(Clone)]
struct Matrix {
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn null() -> Self {
        Self {
            data: vec![0.0; N * N],
        }
    }
    pub fn random() -> Self {
        Self {
            data: (0..N * N).map(|_| rand::random::<f32>()).collect(),
        }
    }
    pub fn print(&self) -> () {
        for i in 0..N {
            println!("{:?}", &self.data[(i * N)..((i + 1) * N)]);
        }
    }
    pub fn eq(&self, other: &Matrix, eps: f32) -> bool {
        for i in 0..N {
            for j in 0..N {
                if (self.data[i * N + j] - other.data[i * N + j]).abs() > eps {
                    return false;
                }
            }
        }
        return true;
    }
}

fn cpu_naive_multiply(a: &Matrix, b: &Matrix, c: &mut Matrix) {
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c.data[i * N + j] += a.data[i * N + k] * b.data[k * N + j];
            }
        }
    }
}

pub struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub kernel: wgpu::ShaderModule,
    pub matrix_a_buffer: wgpu::Buffer,
    pub matrix_b_buffer: wgpu::Buffer,
    pub matrix_c_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
}

async fn init_gpu_state(a: &Matrix, b: &Matrix, c: &Matrix) -> GpuState {
    // Instance contains driver related state for our application.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });

    // Adapter is our "client" holding connection to GPU.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to request adapter");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("GPU example device"),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to request adapter");

    // Create a GPU program.
    let kernel_code = include_str!("shader.wgsl")
        .to_owned()
        .replace("{{N}}", format!("{}", N).as_str())
        .to_owned()
        .replace("{{WORKGROUP_SIZE}}", format!("{}", WORKGROUP_SIZE).as_str());
    let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("GPU compute shader"),
        source: wgpu::ShaderSource::Wgsl(kernel_code.into()),
    });

    // Allocate GPU buffers to store all matrices.
    let matrix_a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix A buffer"),
        contents: bytemuck::cast_slice(&a.data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let matrix_b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix B buffer"),
        contents: bytemuck::cast_slice(&b.data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let matrix_c_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix C buffer"),
        contents: bytemuck::cast_slice(&c.data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging buffer"),
        size: BUFF_SIZE,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Matrix multiply pipeline"),
        layout: None,
        module: &kernel,
        entry_point: "main",
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matrix_c_buffer.as_entire_binding(),
            },
        ],
    });

    return GpuState {
        device,
        queue,
        pipeline,
        bind_group,
        bind_group_layout,
        kernel,
        matrix_a_buffer,
        matrix_b_buffer,
        matrix_c_buffer,
        staging_buffer,
    };
}

async fn gpu_multiply(state: &mut GpuState, a: &Matrix, b: &Matrix, c: &mut Matrix) {
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&state.pipeline);
        cpass.set_bind_group(0, &state.bind_group, &[]);
        cpass.insert_debug_marker("Multiply matrices");
        cpass.dispatch_workgroups((N / WORKGROUP_SIZE) as u32, (N / WORKGROUP_SIZE) as u32, 1);
    }
    encoder.copy_buffer_to_buffer(
        &state.matrix_c_buffer,
        0,
        &state.staging_buffer,
        0,
        BUFF_SIZE,
    );

    state.queue.submit(Some(encoder.finish()));

    let buffer_slice = state.staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

    state.device.poll(wgpu::Maintain::Wait);

    let res = if let Ok(Ok(())) = rx.recv() {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        state.staging_buffer.unmap();
        Some(result)
    } else {
        panic!("Failed to receive data from GPU");
    };

    c.data = res.unwrap();
}

fn main() {
    let mut a = Matrix::random();
    let mut b = Matrix::random();
    let mut c1 = Matrix::null();
    let mut c2 = Matrix::null();

    let mut state = pollster::block_on(init_gpu_state(&a, &b, &c2));

    {
        let start = std::time::Instant::now();
        cpu_naive_multiply(&a, &b, &mut c1);
        println!("CPU naive: {:?}", start.elapsed());
        // c1.print();
    }

    println!();

    {
        let start = std::time::Instant::now();
        pollster::block_on(gpu_multiply(&mut state, &a, &b, &mut c2));
        println!("GPU naive: {:?}", start.elapsed());
        // c2.print();
    }

    println!("Equal: {}", c1.eq(&c2, 0.01));
}
