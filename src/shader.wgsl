struct Matrix {
    data: array<f32>,
};


@group(0) @binding(0) var<storage>        matrix_a:  Matrix;
@group(0) @binding(1) var<storage>        matrix_b:  Matrix;
@group(0) @binding(2) var<storage, write> matrix_c:  Matrix;

@compute
@workgroup_size({{WORKGROUP_SIZE}}, {{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    var sum: f32 = 0.0;
    let N: u32 = {{N}}u;

    for (var i = 0u; i < N; i = i + 1u) {
        let a_i: u32 = i + id.x * N;
        let b_i: u32 = id.y + i * N;

        let a_elem = matrix_a.data[a_i];
        let b_elem = matrix_b.data[b_i];
        sum = sum + a_elem * b_elem;
    }

    let c_i: u32 = id.x * N + id.y;
    matrix_c.data[c_i] = sum;
}
