#version 450
#pragma shader_stage(compute)
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint N = 64;

layout(push_constant) uniform Params
    {
        uint L;
    } params;

layout(std430, column_major, set = 0, binding = 0) buffer restrict readonly VectorV {
    vec2[N] v;
};

layout(std430, column_major, set = 0, binding = 1) buffer restrict readonly VectorZ {
    vec2[] z;
};

layout(std430, column_major, set = 0, binding = 2) buffer restrict readonly VectorQ {
    vec2[N] w;
};

layout(std430, column_major, set = 0, binding = 3) buffer restrict writeonly VectorOutput {
    vec2[] o;
};

void main() {
    if (gl_GlobalInvocationID.y < params.L) {
        vec2 result = vec2(0.0, 0.0);
        vec2 t_z = z[gl_GlobalInvocationID.y];
        [[unroll]] for (int item = 0; item < (N / gl_SubgroupSize); item++) {
            uint N_idx = gl_SubgroupInvocationID + item * gl_SubgroupSize;
            if (N_idx < N) {
                vec2 denominator = (w[N_idx] - t_z);
                result += vec2(dot(v[N_idx].xy, denominator.xy), dot(vec2(1.0, -1.0) * (v[N_idx].yx), denominator.xy)) / dot(denominator, denominator);
            }
        }

        vec2 sum = subgroupAdd(result);
        if (subgroupElect()) {
            o[gl_GlobalInvocationID.y] = sum;
        }
    }
}