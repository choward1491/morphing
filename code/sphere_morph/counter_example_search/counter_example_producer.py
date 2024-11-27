import numpy as np
import sphere_repr as sr

def search_for_counter_examples_convexify(sphere_embedding_generator, num_trials: int, dir_to_save: str = "counter-examples/"):
    assert (num_trials >= 0)
    counter = 0
    bad_inputs = []
    while counter < num_trials:
        sample_embedding: sr.SphereEmbedding = sphere_embedding_generator.sample()
        all_quads = sample_embedding.try_find_all_nonconvex_quads()
        if len(all_quads) == 0:
            continue
        num_fails = 0
        for quad in all_quads:
            reflex_idx, direction = sample_embedding._get_reflex_vertex_and_direction(quad[0])
            reflex_vertex = sample_embedding.polyhedron_repr.get_vertex(quad[0][reflex_idx])
            print("reflex_vertex = ")
            print(reflex_vertex.aux_data.pos)
            if sample_embedding.polyhedron_repr.out_degree(reflex_vertex) > 3:
                succeed, is_strict_convex, output_coordinates, direction, R2 = sample_embedding._convexify_quad2(quad[0], quad[1], quad[2])
                if not succeed or (succeed and not is_strict_convex):
                    if succeed:
                        print("Is not strictly convex")
                    #succeed, is_strict_convex, output_coordinates, direction, R2 = sample_embedding._convexify_quad2(quad[0], quad[1], quad[2])
                    #print("Found one failure!")
                    #print(f"Num coordinates = {len(output_coordinates)}")
                    #print(f"Num vertices = {sample_embedding.polyhedron_repr.num_vertices}")
                    #bad_inputs.append((sample_embedding, quad, output_coordinates, R2))
                    num_fails += 1
                    #break
        if num_fails == len(all_quads):
            bad_inputs.append(sample_embedding)
        counter += 1
    return bad_inputs

def search_for_counter_examples_morph_south_hemisphere(sphere_embedding_generator, num_trials: int, dir_to_save: str = "counter-examples/"):
    assert (num_trials >= 0)
    counter = 0
    bad_inputs = []
    while counter < num_trials:
        sample_embedding: sr.SphereEmbedding = sphere_embedding_generator.sample()
        num_vertices = sample_embedding.polyhedron_repr.num_vertices
        num_succeed = 0
        bad_north_poles = []
        for idx in range(0, num_vertices):
            (did_succeed, new_coordinates) = sample_embedding.try_morph_into_southern_hemisphere(north_pole_idx=idx)
            if did_succeed:
                num_succeed += 1
            else:
                bad_north_poles.append(idx)
        if num_succeed == 0:
            print("Found bad input for morphing to southern hemisphere!")
            bad_inputs.append((bad_north_poles, sample_embedding))
        counter += 1
    return bad_inputs