//! Cross-map BSP3D geometry invariants: floor/ceiling normals & winding,
//! degenerate polygons, flat coplanarity, generated blockmap coverage.

use test_utils::{
    assert_floor_ceiling_normals, doom_wad_path, doom1_wad_path, load_map, load_map_with_pwad,
    sigil2_wad_path,
};

// ---------------------------------------------------------------------------
// Floor/ceiling normals & winding (every non-sky subsector: one +Z floor, one
// −Z ceiling, smaller XY ⊆ larger). Shared scan in `test_utils`.
// ---------------------------------------------------------------------------

#[test]
fn e1m1_floor_ceiling_normals() {
    assert_floor_ceiling_normals(&load_map(&doom1_wad_path(), "E1M1"));
}

#[cfg_attr(not(feature = "wad-doom"), ignore = "needs doom.wad (~/doom/)")]
#[test]
fn e1m2_floor_ceiling_normals() {
    assert_floor_ceiling_normals(&load_map(&doom_wad_path(), "E1M2"));
}

#[cfg_attr(
    all(not(feature = "wad-doom"), not(feature = "wad-sigil2")),
    ignore = "needs doom.wad + sigil2.wad (~/doom/)"
)]
#[test]
fn e6m1_floor_ceiling_normals() {
    let map = load_map_with_pwad(&doom_wad_path(), &sigil2_wad_path(), "E6M1");
    assert_floor_ceiling_normals(&map);
}

// ---------------------------------------------------------------------------
// Degenerate-polygon scans: no floor/ceiling polygon may have < 3 vertices,
// duplicate indices, or zero-length edges.
// ---------------------------------------------------------------------------

#[cfg_attr(not(feature = "wad-doom"), ignore = "needs doom.wad (~/doom/)")]
#[test]
fn e1m2_no_degenerate_polygons() {
    let map = load_map(&doom_wad_path(), "E1M2");
    let bsp3d = &map.bsp_3d;
    let verts = &bsp3d.vertices;
    let mut failures = Vec::new();

    for (ssid, leaf) in bsp3d.subsector_leaves.iter().enumerate() {
        for (label, indices) in [
            ("floor", &leaf.floor_polygons),
            ("ceil", &leaf.ceiling_polygons),
        ] {
            for &pi in indices {
                let poly = &bsp3d.polygons[pi];
                let n = poly.vertices.len();
                if n < 3 {
                    failures.push(format!("ss={ssid} {label}: < 3 vertices ({n})"));
                    continue;
                }
                for i in 0..n {
                    for j in (i + 1)..n {
                        if poly.vertices[i] == poly.vertices[j] {
                            failures.push(format!(
                                "ss={ssid} {label}: duplicate index {} at {i},{j}",
                                poly.vertices[i]
                            ));
                        }
                    }
                }
                for i in 0..n {
                    let a = verts[poly.vertices[i]];
                    let b = verts[poly.vertices[(i + 1) % n]];
                    let dist = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
                    if dist < 0.01 {
                        failures.push(format!(
                            "ss={ssid} {label}: zero-length edge [{i}<->{}] dist={dist:.6}",
                            (i + 1) % n
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[cfg_attr(
    all(not(feature = "wad-doom"), not(feature = "wad-sigil2")),
    ignore = "needs doom.wad + sigil2.wad (~/doom/)"
)]
#[test]
fn e6m1_no_degenerate_floor_polygons() {
    let map = load_map_with_pwad(&doom_wad_path(), &sigil2_wad_path(), "E6M1");
    let bsp3d = &map.bsp_3d;
    let verts = &bsp3d.vertices;
    let mut failures = Vec::new();

    for (ssid, leaf) in bsp3d.subsector_leaves.iter().enumerate() {
        for &fp_idx in &leaf.floor_polygons {
            let poly = &bsp3d.polygons[fp_idx];
            let n = poly.vertices.len();
            if n < 3 {
                failures.push(format!("ss={ssid} fp={fp_idx}: < 3 vertices"));
                continue;
            }
            let has_dup =
                (0..n).any(|i| ((i + 1)..n).any(|j| poly.vertices[i] == poly.vertices[j]));
            if has_dup {
                failures.push(format!("ss={ssid} fp={fp_idx}: duplicate vertex index"));
                continue;
            }
            let area: f32 = (0..n)
                .map(|i| {
                    let a = verts[poly.vertices[i]];
                    let b = verts[poly.vertices[(i + 1) % n]];
                    a.x * b.y - b.x * a.y
                })
                .sum();
            if area <= 0.0 {
                failures.push(format!(
                    "ss={ssid} fp={fp_idx}: shoelace={area:.2} (expected > 0)"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Flat coplanarity: every vertex in a floor (or ceiling) polygon shares one Z.
// ---------------------------------------------------------------------------

#[cfg_attr(not(feature = "wad-doom"), ignore = "needs doom.wad (~/doom/)")]
#[test]
fn e1m2_flat_polygon_coplanarity() {
    let map = load_map(&doom_wad_path(), "E1M2");
    let bsp3d = &map.bsp_3d;
    let verts = &bsp3d.vertices;
    let mut failures = Vec::new();

    for (ssid, leaf) in bsp3d.subsector_leaves.iter().enumerate() {
        for (label, indices) in [
            ("floor", &leaf.floor_polygons),
            ("ceil", &leaf.ceiling_polygons),
        ] {
            for &pi in indices {
                let poly = &bsp3d.polygons[pi];
                if poly.vertices.is_empty() {
                    continue;
                }
                let z0 = verts[poly.vertices[0]].z;
                for &vi in &poly.vertices[1..] {
                    let z = verts[vi].z;
                    if (z - z0).abs() > 0.01 {
                        failures.push(format!(
                            "ss={ssid} {label}: vertex {vi} z={z:.2} != {z0:.2}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Specific-subsector floor polygon validity (E6M1 ss2587 regression).
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(not(feature = "wad-doom"), not(feature = "wad-sigil2")),
    ignore = "needs doom.wad + sigil2.wad (~/doom/)"
)]
#[test]
fn e6m1_subsector_2587_polygon() {
    let map = load_map_with_pwad(&doom_wad_path(), &sigil2_wad_path(), "E6M1");
    let bsp3d = &map.bsp_3d;
    let verts = &bsp3d.vertices;

    assert!(
        bsp3d.subsector_leaves.len() > 2587,
        "map must have >= 2588 subsectors, got {}",
        bsp3d.subsector_leaves.len()
    );

    let leaf = &bsp3d.subsector_leaves[2587];
    for &fp_idx in &leaf.floor_polygons {
        let poly = &bsp3d.polygons[fp_idx];
        let n = poly.vertices.len();
        assert!(n >= 3, "floor polygon must have >= 3 vertices, got {n}");
        for i in 0..n {
            for j in (i + 1)..n {
                assert_ne!(
                    poly.vertices[i], poly.vertices[j],
                    "duplicate vertex index {} at {i},{j}",
                    poly.vertices[i]
                );
            }
        }
        let area: f32 = (0..n)
            .map(|i| {
                let a = verts[poly.vertices[i]];
                let b = verts[poly.vertices[(i + 1) % n]];
                a.x * b.y - b.x * a.y
            })
            .sum();
        assert!(area > 0.0, "floor shoelace must be positive, got {area}");
    }
}

// ---------------------------------------------------------------------------
// Generated blockmap covers the same grid as the WAD's and every linedef.
// ---------------------------------------------------------------------------

#[test]
fn e1m1_generated_blockmap_coverage() {
    let mut map = load_map(&doom1_wad_path(), "E1M1");
    let wad_bm = map.blockmap();
    let (wad_cols, wad_rows) = (wad_bm.columns, wad_bm.rows);
    assert!(
        wad_cols > 0 && wad_rows > 0,
        "WAD blockmap should exist for E1M1"
    );

    map.build_blockmap("E1M1");
    let gen_bm = map.blockmap();
    assert_eq!(
        gen_bm.columns, wad_cols,
        "generated columns should match WAD"
    );
    assert_eq!(gen_bm.rows, wad_rows, "generated rows should match WAD");
    assert!(
        !gen_bm.block_lines.is_empty(),
        "generated blockmap should have line refs"
    );

    let num_lines = map.linedefs.len();
    let mut line_found = vec![false; num_lines];
    for i in 0..gen_bm.block_offsets.len() - 1 {
        for j in gen_bm.block_offsets[i]..gen_bm.block_offsets[i + 1] {
            let ld_num = gen_bm.block_lines[j].num;
            if ld_num < num_lines {
                line_found[ld_num] = true;
            }
        }
    }
    let missing: Vec<usize> = line_found
        .iter()
        .enumerate()
        .filter(|(_, found)| !**found)
        .map(|(i, _)| i)
        .collect();
    assert!(
        missing.is_empty(),
        "every linedef should appear in a blockmap cell; missing: {:?}",
        &missing[..missing.len().min(10)]
    );
}

#[cfg_attr(not(feature = "wad-sunder"), ignore = "needs sunder.wad (~/doom/)")]
#[test]
fn sunder_map20_generated_blockmap() {
    use test_utils::sunder_wad_path;
    let map = load_map(&sunder_wad_path(), "MAP20");
    let bm = map.blockmap();
    assert!(
        bm.columns > 0 && bm.rows > 0,
        "blockmap should have valid dimensions"
    );
    assert!(!bm.block_lines.is_empty(), "blockmap should have line refs");
    let total = bm.columns * bm.rows;
    assert!(
        total > 1000,
        "MAP20 blockmap should be large, got {}x{}",
        bm.columns,
        bm.rows
    );
}
