	.text
	.globl	infer                   // -- Begin function infer
	.p2align	3
infer:                                  // @infer
// %bb.0:
	dstcr	0, rp0
	dstcr	0, rp1
	dstcr	-272, r10
	stcr	0, crp0
	stcr	0, crp1
	dstcr	0x0, pc.mode, east
	dstcr	0x0, pc.mode, north
	dstcr	0x0, pc.mode, south
	dstcr	0x0, pc.mode, west
	dstcr	0x1, pc.resetfifo, east
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x1, pc.resetfifo, south
	dstcr	0x1, pc.resetfifo, west
	daddi32	rp1, r10, rp1
	dstcr	256, rp2
	dcp	p5, r8
	daddi32	rp1, rp2, rp2
	dcp	p4, r9
	dcp	p3, r10
	dcp	p2, r11
	dcp	p1, r18
	dcp	p0, r19
	dstcr	0, r12
	daddi32	rp2, 4, rp2
	dstcr	0, r13
	addi32	crp1, -40, crp1         //     
	//APP
	nop <> __iss__ profile start -msg "Layer_0_fused_transpose_14"
	//NO_APP
	dstcr	0x2, mode
.LBB0_1:                                // %loadstoreloop160
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r12, 1, r14
	dstcr	0, [rp2+=1]
	dcmplt32	r14, r12, r12
	dcmplt32	r14, 3, r15
	daddi32	r13, r12, r13
	dcp	r14, r12
	dcmpeq32	r13, 0, r14
	dcmpneq32	r14, 0, r14
	dcsel	r15, 0, r14
	djmpneqoff	r14, 0, :.LBB0_1
// %bb.2:                               // %split159
	daddi32	rp1, 240, rp2
	dstcr	256, rp3
	dstcr	0, r13
	daddi32	rp2, 4, rp2
	dstcr	0, r12
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_3:                                // %loadstoreloop158
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r13, 1, r14
	dstcr	0, [rp2+=1]
	dcmplt32	r14, r13, r13
	dcmplt32	r14, 3, r15
	daddi32	r12, r13, r12
	dcp	r14, r13
	dcmpeq32	r12, 0, r14
	dcmpneq32	r14, 0, r14
	dcsel	r15, 0, r14
	djmpneqoff	r14, 0, :.LBB0_3
// %bb.4:                               // %split157
	dstcr	535232, r12
	daddi32	rp1, 240, rp2
	daddi32	r19, r12, r12
	dstcr	4753408, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r12, r19, r13
	dstcr	0x488800, els.intaddr
	dcp	r12, els.extaddrl
	daddi32	r18, r13, r12
	dcp	r12, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x380, els.intstride2
	dstcr	0xe, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x380, els.extstride2
	dstcr	0xe, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_5:                                // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r12
	djmpneqoff	r12, 0, :.LBB0_5
// %bb.6:
	daddi32	rp1, 244, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	cp	row, cr10
	cp	col, cr11
	dstcr	0x0, dependencyid
	dstcr	0x488b80, els.intaddr
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x82000, els.intstride2
	dstcr	0x2080, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x82000, els.extstride2
	dstcr	0x2080, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_7:                                // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_7
// %bb.8:
	muli32	cr11, 3, cr11
	dstcr	0x1, elsstatus
	dstcr	0, r10
	dstcr	448, r11
	dstcr	4194304, r12
	stcr	1280, cr12
	addi32	crp1, 8, crp2           //      
	stcr	416, cr13
	dstcr	33686018, r13
	stcr	32792, cr14
	dstcr	186368, r14
	dstcr	0, r15
	dcp	flowid, r16
	dstcr	0x10, pls.count1, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	stcr	0x0, bitwidthmode
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x1c0, pls.stride1, north
	dstcr	0x10, pls.stride2, north
.LBB0_9:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_10 Depth 2
                                        //       Child Loop BB0_11 Depth 3
                                        //         Child Loop BB0_13 Depth 4
                                        //         Child Loop BB0_15 Depth 4
                                        //         Child Loop BB0_17 Depth 4
                                        //         Child Loop BB0_19 Depth 4
                                        //         Child Loop BB0_21 Depth 4
                                        //       Child Loop BB0_25 Depth 3
                                        //         Child Loop BB0_26 Depth 4
                                        //         Child Loop BB0_28 Depth 4
	dshlb	r15, 4, r6
	dmuli32	r10, r11, r5
	dcpc	r10, cr15
	dcpc	r6, cr16
	addi32	cr16, cr10, cr16
	addi32	cr15, cr10, cr15
	muli32	cr16, cr12, cr16
	dstcr	0, r17
	daddi32	r5, r12, r5
	addi32	cr16, cr11, cr16
.LBB0_10:                               //   Parent Loop BB0_9 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_11 Depth 3
                                        //         Child Loop BB0_13 Depth 4
                                        //         Child Loop BB0_15 Depth 4
                                        //         Child Loop BB0_17 Depth 4
                                        //         Child Loop BB0_19 Depth 4
                                        //         Child Loop BB0_21 Depth 4
                                        //       Child Loop BB0_25 Depth 3
                                        //         Child Loop BB0_26 Depth 4
                                        //         Child Loop BB0_28 Depth 4
	dcpc	r17, cr17
	addi32	cr16, cr17, cr17
	dstcr	0, r6
	cp	crp2, crp3
	dstcr	0, r7
	dstcr	0x488b80, pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x1, pls.count2, north
.LBB0_11:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB0_13 Depth 4
                                        //         Child Loop BB0_15 Depth 4
                                        //         Child Loop BB0_17 Depth 4
                                        //         Child Loop BB0_19 Depth 4
                                        //         Child Loop BB0_21 Depth 4
	dcpc	r6, cr5
	addi32	cr17, cr5, cr5
	cmplt32	cr15, cr13, cr7
	stcr	0, cr6
	predpush	cr7, :.LBB0_23
// %bb.12:                              //   in Loop: Header=BB0_11 Depth=3
	dstcr	0x5, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	r16, dependencyid
.LBB0_13:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        //       Parent Loop BB0_11 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	dandb	r13, plsstatus, r28
	dorb	r28, pelsr, r28
	djmpneqoff	r28, 0, :.LBB0_13
// %bb.14:                              //   in Loop: Header=BB0_11 Depth=3
	djmpincsetup	0, 4, :.LBB0_15
	dstcr	0x1, plsstatus, north
	nrb	cr5, north
	dstcr	0x260, pc.mode, north
.LBB0_15:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        //       Parent Loop BB0_11 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB0_11 Depth=3
	djmpincsetup	0, 16, :.LBB0_17
	dstcr	0x360, pc.mode, north
.LBB0_17:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        //       Parent Loop BB0_11 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB0_11 Depth=3
	djmpincsetup	0, 16, :.LBB0_19
.LBB0_19:                               // %.preheader
                                        //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        //       Parent Loop BB0_11 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	north, south
// %bb.20:                              //   in Loop: Header=BB0_11 Depth=3
	djmpincsetup	0, 4, :.LBB0_21
	dstcr	0x260, pc.mode, north
.LBB0_21:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        //       Parent Loop BB0_11 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	north, south
// %bb.22:                              //   in Loop: Header=BB0_11 Depth=3
	shlb	north.0z, 8, cr6
.LBB0_23:                               // %Flow338
                                        //   in Loop: Header=BB0_11 Depth=3
	predpop	
	divi32	cr6, 255, cr5
	daddi32	r6, 48, r6
	muli32lohi{8}	cr5, cr14, cr5
	shrab	cr5, 8, cr5
	mini32	cr5, 127, cr5
	maxi32	cr5, -127, [crp3.z+=1]
	djmpincne	r7, 26, :.LBB0_11
// %bb.24:                              //   in Loop: Header=BB0_10 Depth=2
	dmuli32	r17, r14, r7
	cp	crp2, crp3
	dstcr	0, r6
	daddi32	r5, r7, r7
	dstcr	0x0, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r7, pls.addr, north
	dstcr	0x1a, pls.count2, north
	dstcr	0x0, dependencyid
	dstcr	0x1, plsstatus, north
.LBB0_25:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB0_26 Depth 4
                                        //         Child Loop BB0_28 Depth 4
	nrb	[crp3.z], north
	djmpincsetup	0, 4, :.LBB0_26
	dstcr	0x200, pc.mode, north
.LBB0_26:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        //       Parent Loop BB0_25 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.27:                              //   in Loop: Header=BB0_25 Depth=3
	djmpincsetup	0, 16, :.LBB0_28
	dstcr	0x300, pc.mode, north
.LBB0_28:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        //       Parent Loop BB0_25 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.29:                              //   in Loop: Header=BB0_25 Depth=3
	addi32	crp3, 1, crp3
	djmpincne	r6, 26, :.LBB0_25
// %bb.30:                              //   in Loop: Header=BB0_10 Depth=2
	dstcr	0x200, pc.mode, north
	djmpincne	r17, 3, :.LBB0_10
// %bb.31:                              //   in Loop: Header=BB0_9 Depth=1
	daddi32	r10, 16, r10
	djmpincne	r15, 26, :.LBB0_9
// %bb.32:
	dstcr	0, r10
	daddi32	rp1, 224, rp2
	dstcr	0, r11
	//APP
	nop <> __iss__ profile stop -msg "Layer_0_fused_transpose_14"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_1_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__4"
	//NO_APP
	dstcr	0x2, mode
.LBB0_33:                               // %loadstoreloop156
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 4, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_33
// %bb.34:                              // %split155
	daddi32	rp1, 208, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_35:                               // %loadstoreloop154
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_35
// %bb.36:                              // %split153
	dstcr	536128, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	692224, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0xa9000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1400, els.intstride2
	dstcr	0x50, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1400, els.extstride2
	dstcr	0x50, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_37:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_37
// %bb.38:
	daddi32	rp1, 212, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 172, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	daddi32	rp1, 224, rp4
	daddi32	rp1, rp2, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__4I10FixedPointIsLh6ELh2ELi0EE7_TensorIaL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj3ELj416ELj416EEES2_IaLS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj896EEES2_IDv4_aLS3_0EjLj64ELS4_1EJLj1ELj4ELj208ELj208EEEEvRT0_RT1_RT2_
	daddi32	rp1, 172, rp2
	dstcr	0x2, mode
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	144 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_1_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__4"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_2_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__3"
	//NO_APP
	dstcr	0x2, mode
.LBB0_39:                               // %loadstoreloop152
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_39
// %bb.40:                              // %split151
	daddi32	rp1, 240, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_41:                               // %loadstoreloop150
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_41
// %bb.42:                              // %split149
	dstcr	541248, r10
	daddi32	rp1, 240, rp2
	daddi32	r19, r10, r10
	dstcr	4567040, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x45b000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x4c00, els.intstride2
	dstcr	0x130, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x4c00, els.extstride2
	dstcr	0x130, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_43:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_43
// %bb.44:
	daddi32	rp1, 244, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 168, rp2
	dstcr	256, rp4
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	daddi32	rp1, 208, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj4ELj208ELj208EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj5120EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj8ELj104ELj104EEEEvRT0_RT1_RT2_
	daddi32	rp1, 168, rp2
	dstcr	0x2, mode
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	163 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	daddi32	rp1, 224, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 228, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_2_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__3"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_3_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__2"
	//NO_APP
	dstcr	0x2, mode
.LBB0_45:                               // %loadstoreloop148
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_45
// %bb.46:                              // %split147
	daddi32	rp1, 208, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_47:                               // %loadstoreloop146
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_47
// %bb.48:                              // %split145
	dstcr	560704, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	212992, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x34000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x12800, els.intstride2
	dstcr	0x4a0, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x12800, els.extstride2
	dstcr	0x4a0, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_49:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_49
// %bb.50:
	daddi32	rp1, 212, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 164, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	daddi32	rp1, 224, rp4
	daddi32	rp1, rp2, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj8ELj104ELj104EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj19456EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj16ELj52ELj52EEEEvRT0_RT1_RT2_
	daddi32	rp1, 164, rp2
	dstcr	0x2, mode
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	182 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_3_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__2"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_4_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__1"
	//NO_APP
	dstcr	0x2, mode
.LBB0_51:                               // %loadstoreloop144
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_51
// %bb.52:                              // %split143
	daddi32	rp1, 240, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_53:                               // %loadstoreloop142
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_53
// %bb.54:                              // %split141
	dstcr	636480, r10
	daddi32	rp1, 240, rp2
	daddi32	r19, r10, r10
	dstcr	4300800, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x41a000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x49000, els.intstride2
	dstcr	0x1240, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x49000, els.extstride2
	dstcr	0x1240, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_55:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_55
// %bb.56:
	daddi32	rp1, 244, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 160, rp2
	dstcr	256, rp4
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	daddi32	rp1, 208, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj16ELj52ELj52EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj75776EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj32ELj26ELj26EEEEvRT0_RT1_RT2_
	daddi32	rp1, 160, rp2
	dstcr	0x2, mode
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	201 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	daddi32	rp1, 224, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 228, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_4_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__1"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_5_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_8552357946429664381_"
	//NO_APP
	dstcr	0x2, mode
.LBB0_57:                               // %loadstoreloop140
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_57
// %bb.58:                              // %split139
	daddi32	rp1, 156, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	daddi32	rp1, 224, rp4
	daddi32	rp1, rp2, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_8552357946429664381_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj32ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj299008EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj256ELj26ELj26EEEEvRT0_RT1_RT2_
	daddi32	rp1, 156, rp2
	dstcr	0x2, mode
	dstcr	10712896, r11
	dcp	[rp2], link
	daddi32	rp1, 228, rp2
	nop <> __iss__ print	214 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 228, rp2
	dshrab	r10, 31, r10
	daddi32	r19, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r19, r12
	daddi32	rp1, 224, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r18, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0xd00, els.intstride2
	dstcr	0x3400, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0xd00, els.extstride2
	dstcr	0x3400, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_59:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_59
// %bb.60:
	daddi32	rp1, 228, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	10286912, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	daddi32	r19, r10, r20
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	dcmplt32	r20, r19, r10
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, r11
	daddi32	r18, r10, r21
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_5_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_8552357946429664381_"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_6_fused_nn_max_pool2d_fixed_point_multiply_cast_round_clip_cast"
	//NO_APP
	dstcr	0x2, mode
.LBB0_61:                               // %loadstoreloop138
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_61
// %bb.62:                              // %split137
	daddi32	rp1, 240, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_63:                               // %loadstoreloop136
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_63
// %bb.64:                              // %split135
	dstcr	935488, r10
	daddi32	rp1, 240, rp2
	daddi32	r19, r10, r10
	dstcr	4247552, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x40d000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x91000, els.intstride2
	dstcr	0x2440, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x91000, els.extstride2
	dstcr	0x2440, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_65:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_65
// %bb.66:
	daddi32	rp1, 244, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 152, rp2
	dstcr	256, rp3
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	daddi32	rp1, rp3, rp3
	dcp	rp2, r10
	dcp	rp3, r11
	djal	:_Z61fused_nn_max_pool2d_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_
	daddi32	rp1, 152, rp2
	dstcr	0x2, mode
	daddi32	rp1, 228, rp3
	dstcr	0, r11
	dcp	[rp2], link
	daddi32	rp1, 224, rp2
	dstcr	0, r10
	nop <> __iss__ print	237 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_6_fused_nn_max_pool2d_fixed_point_multiply_cast_round_clip_cast"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_7_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_"
	//NO_APP
	dstcr	0x2, mode
.LBB0_67:                               // %loadstoreloop134
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_67
// %bb.68:                              // %split133
	daddi32	rp1, 208, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_69:                               // %loadstoreloop132
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_69
// %bb.70:                              // %split131
	dstcr	1529408, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	53248, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0xd000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x91000, els.intstride2
	dstcr	0x2440, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x91000, els.extstride2
	dstcr	0x2440, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_71:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_71
// %bb.72:
	daddi32	rp1, 212, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 148, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	daddi32	rp1, 224, rp4
	daddi32	rp1, rp2, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 148, rp2
	dstcr	0x2, mode
	dstcr	9459520, r11
	dcp	[rp2], link
	daddi32	rp1, 228, rp2
	nop <> __iss__ print	255 -dec 
	dshlb	[rp2], 16, r10
	daddi32	r19, r11, r22
	dshrab	r10, 31, r10
	daddi32	rp1, 228, rp2
	dcmplt32	r22, r19, r11
	dandb	[rp2], r10, r10
	daddi32	rp1, 224, rp2
	dcp	r10, dependencyid
	daddi32	r18, r11, r23
	dcp	[rp2], els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x340, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x340, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_73:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_73
// %bb.74:
	daddi32	rp1, 228, rp3
	daddi32	rp1, 240, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_7_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_8_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_"
	//NO_APP
	dstcr	0x2, mode
.LBB0_75:                               // %loadstoreloop130
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_75
// %bb.76:                              // %split129
	daddi32	rp1, 240, rp2
	daddi32	rp1, 208, rp3
	dstcr	4247552, [rp2]
	daddi32	rp1, 144, rp2
	daddi32	rp1, 240, rp4
	dcp	link, [rp2]
	dstcr	256, rp2
	dcp	rp3, r11
	daddi32	rp1, rp2, rp2
	dcp	rp4, r12
	dcp	rp2, r10
	djal	:_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 144, rp2
	dstcr	0x2, mode
	dstcr	9512768, r11
	dcp	[rp2], link
	daddi32	rp1, 244, rp2
	nop <> __iss__ print	271 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 244, rp2
	dshrab	r10, 31, r10
	daddi32	r19, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r19, r12
	daddi32	rp1, 240, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r18, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x340, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x340, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_77:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_77
// %bb.78:
	daddi32	rp1, 244, rp3
	daddi32	rp1, 208, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_8_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_9_fused_concatenate_9"
	//NO_APP
	dstcr	0x2, mode
.LBB0_79:                               // %loadstoreloop128
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_79
// %bb.80:                              // %split127
	daddi32	rp1, 192, rp2
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, 208, rp3
	dstcr	53248, [rp3]
.LBB0_81:                               // %loadstoreloop126
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_81
// %bb.82:                              // %split125
	daddi32	rp1, 192, rp2
	dstcr	159744, [rp2]
	dstcr	0x0, dependencyid
	dstcr	0xd000, els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x680, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x680, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_83:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_83
// %bb.84:
	dstcr	2123328, r10
	daddi32	rp1, 212, rp2
	daddi32	r19, r10, r10
	dstcr	0x1, elsstatus
	dcmplt32	r10, r19, r11
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x27000, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x242000, els.intstride2
	dstcr	0x9080, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x242000, els.extstride2
	dstcr	0x9080, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_85:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_85
// %bb.86:
	daddi32	rp1, 196, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	dstcr	0, r11
	daddi32	rp1, rp3, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_9_fused_concatenate_9"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_10_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2"
	//NO_APP
	dstcr	0x2, mode
.LBB0_87:                               // %loadstoreloop124
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_87
// %bb.88:                              // %split123
	daddi32	rp1, 240, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_89:                               // %loadstoreloop122
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_89
// %bb.90:                              // %split121
	dstcr	4490816, r10
	daddi32	rp1, 240, rp2
	daddi32	r19, r10, r10
	dstcr	4300800, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x41a000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x242000, els.intstride2
	dstcr	0x9080, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x242000, els.extstride2
	dstcr	0x9080, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_91:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_91
// %bb.92:
	daddi32	rp1, 244, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 140, rp2
	dstcr	256, rp4
	dcp	link, [rp2]
	daddi32	rp1, 208, rp2
	daddi32	rp1, 192, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj2367488EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 140, rp2
	dstcr	0x2, mode
	dstcr	9566016, r11
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	314 -dec 
	daddi32	rp1, rp2, rp2
	daddi32	r19, r11, r22
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dcmplt32	r22, r19, r11
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	daddi32	r18, r11, r23
	dcp	[rp2], els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x680, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x680, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_93:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_93
// %bb.94:
	dstcr	260, rp3
	daddi32	rp1, 192, rp2
	daddi32	rp1, rp3, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 196, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_10_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_11_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2"
	//NO_APP
	dstcr	0x2, mode
.LBB0_95:                               // %loadstoreloop120
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_95
// %bb.96:                              // %split119
	daddi32	rp1, 192, rp2
	daddi32	rp1, 240, rp3
	dstcr	159744, [rp2]
	daddi32	rp1, 136, rp2
	daddi32	rp1, 192, rp4
	dcp	link, [rp2]
	daddi32	rp1, 208, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj2367488EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 136, rp2
	dstcr	0x2, mode
	dstcr	9672512, r11
	dcp	[rp2], link
	daddi32	rp1, 196, rp2
	nop <> __iss__ print	330 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 196, rp2
	dshrab	r10, 31, r10
	daddi32	r19, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r19, r12
	daddi32	rp1, 192, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r18, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x680, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x680, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_97:                               // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_97
// %bb.98:
	daddi32	rp1, 196, rp3
	daddi32	rp1, 240, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_11_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_12_fused_concatenate_10"
	//NO_APP
	dstcr	0x2, mode
.LBB0_99:                               // %loadstoreloop118
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_99
// %bb.100:                             // %split117
	daddi32	rp1, 176, rp2
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, 240, rp3
	dstcr	4300800, [rp3]
.LBB0_101:                              // %loadstoreloop116
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_101
// %bb.102:                             // %split115
	daddi32	rp1, 176, rp2
	dstcr	4513792, [rp2]
	dstcr	0x0, dependencyid
	dstcr	0x41a000, els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0xd00, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0xd00, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_103:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_103
// %bb.104:
	dstcr	6858304, r10
	daddi32	rp1, 244, rp2
	daddi32	r19, r10, r10
	dstcr	0x1, elsstatus
	dcmplt32	r10, r19, r11
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x44e000, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x41000, els.intstride2
	dstcr	0x1040, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x41000, els.extstride2
	dstcr	0x1040, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_105:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_105
// %bb.106:
	daddi32	rp1, 180, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 196, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	daddi32	rp1, 224, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	0, [rp3]
	daddi32	rp1, 228, rp3
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_12_fused_concatenate_10"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_13_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__1"
	//NO_APP
	dstcr	0x2, mode
.LBB0_107:                              // %loadstoreloop114
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_107
// %bb.108:                             // %split113
	daddi32	rp1, 208, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_109:                              // %loadstoreloop112
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_109
// %bb.110:                             // %split111
	dstcr	7124544, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	53248, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0xd000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x91000, els.intstride2
	dstcr	0x2440, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x91000, els.extstride2
	dstcr	0x2440, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_111:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_111
// %bb.112:
	daddi32	rp1, 212, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 132, rp2
	daddi32	rp1, 176, rp3
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	daddi32	rp1, 224, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj266240EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_RT2_
	daddi32	rp1, 132, rp2
	dstcr	0x2, mode
	dstcr	9779008, r11
	dcp	[rp2], link
	daddi32	rp1, 228, rp2
	nop <> __iss__ print	374 -dec 
	dshlb	[rp2], 16, r10
	daddi32	r19, r11, r24
	dshrab	r10, 31, r10
	daddi32	rp1, 228, rp2
	dcmplt32	r24, r19, r11
	dandb	[rp2], r10, r10
	daddi32	rp1, 224, rp2
	dcp	r10, dependencyid
	daddi32	r18, r11, r25
	dcp	[rp2], els.intaddr
	dcp	r24, els.extaddrl
	dcp	r25, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x340, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x340, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_113:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_113
// %bb.114:
	daddi32	rp1, 228, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 180, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 176, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_13_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__1"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_14_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_"
	//NO_APP
	dstcr	0x2, mode
.LBB0_115:                              // %loadstoreloop110
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_115
// %bb.116:                             // %split109
	daddi32	rp1, 240, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_117:                              // %loadstoreloop108
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_117
// %bb.118:                             // %split107
	dstcr	7718464, r10
	daddi32	rp1, 240, rp2
	daddi32	r19, r10, r10
	dstcr	4247552, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x40d000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x91000, els.intstride2
	dstcr	0x2440, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x91000, els.extstride2
	dstcr	0x2440, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_119:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_119
// %bb.120:
	daddi32	rp1, 244, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 128, rp2
	dstcr	256, rp4
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	daddi32	rp1, 208, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 128, rp2
	dstcr	0x2, mode
	dstcr	9832256, r11
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	398 -dec 
	daddi32	rp1, rp2, rp2
	daddi32	r19, r11, r22
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dcmplt32	r22, r19, r11
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	daddi32	r18, r11, r23
	dcp	[rp2], els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x340, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x340, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_121:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_121
// %bb.122:
	dstcr	260, rp3
	daddi32	rp1, 208, rp2
	daddi32	rp1, rp3, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_14_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_15_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_"
	//NO_APP
	dstcr	0x2, mode
.LBB0_123:                              // %loadstoreloop106
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_123
// %bb.124:                             // %split105
	daddi32	rp1, 208, rp2
	daddi32	rp1, 240, rp3
	dstcr	53248, [rp2]
	daddi32	rp1, 124, rp2
	daddi32	rp1, 208, rp4
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 124, rp2
	dstcr	0x2, mode
	dstcr	9885504, r11
	dcp	[rp2], link
	daddi32	rp1, 212, rp2
	nop <> __iss__ print	414 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 212, rp2
	dshrab	r10, 31, r10
	daddi32	r19, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r19, r12
	daddi32	rp1, 208, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r18, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x340, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x340, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_125:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_125
// %bb.126:
	daddi32	rp1, 212, rp3
	daddi32	rp1, 240, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_15_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_16_fused_concatenate_9"
	//NO_APP
	dstcr	0x2, mode
.LBB0_127:                              // %loadstoreloop104
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_127
// %bb.128:                             // %split103
	daddi32	rp1, 192, rp2
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, 240, rp3
	dstcr	4247552, [rp3]
.LBB0_129:                              // %loadstoreloop102
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_129
// %bb.130:                             // %split101
	daddi32	rp1, 192, rp2
	dstcr	4354048, [rp2]
	dstcr	0x0, dependencyid
	dstcr	0x40d000, els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x680, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x680, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_131:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_131
// %bb.132:
	dstcr	8312384, r10
	daddi32	rp1, 244, rp2
	daddi32	r19, r10, r10
	dstcr	0x1, elsstatus
	dcmplt32	r10, r19, r11
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x427000, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x20600, els.intstride2
	dstcr	0x818, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x20600, els.extstride2
	dstcr	0x818, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_133:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_133
// %bb.134:
	daddi32	rp1, 196, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	daddi32	rp1, 208, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 228, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_16_fused_concatenate_9"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_17_fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943_"
	//NO_APP
	dstcr	0x2, mode
.LBB0_135:                              // %loadstoreloop100
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_135
// %bb.136:                             // %split99
	daddi32	rp1, 120, rp2
	daddi32	rp1, 192, rp3
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	daddi32	rp1, 208, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z102fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj132600EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj13ELj13EEEEvRT0_RT1_RT2_
	daddi32	rp1, 120, rp2
	dstcr	0x2, mode
	daddi32	rp1, 196, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	451 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 244, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_17_fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943_"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_18_fused_reshape_14"
	//NO_APP
	dstcr	0x2, mode
.LBB0_137:                              // %loadstoreloop98
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_137
// %bb.138:                             // %split97
	dstcr	256, rp2
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 116, rp2
	dcp	rp3, r11
	dcp	link, [rp2]
	daddi32	rp1, 208, rp2
	dcp	rp2, r10
	djal	:_Z16fused_reshape_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj13ELj13EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj169EEEEvRT0_RT1_
	daddi32	rp1, 116, rp2
	dstcr	0x2, mode
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	dcp	[rp2], link
	daddi32	rp1, 240, rp2
	dstcr	0, r10
	nop <> __iss__ print	465 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_18_fused_reshape_14"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_19_fused_transpose_13"
	//NO_APP
	dstcr	0x2, mode
.LBB0_139:                              // %loadstoreloop96
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_139
// %bb.140:                             // %split95
	daddi32	rp1, 112, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	dcp	rp3, r11
	daddi32	rp1, rp2, rp2
	dcp	rp2, r10
	djal	:_Z18fused_transpose_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj169EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj169ELj1ELj255EEEEvRT0_RT1_
	daddi32	rp1, 112, rp2
	dstcr	0x2, mode
	dstcr	260, rp3
	dstcr	0, r11
	dcp	[rp2], link
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	nop <> __iss__ print	477 -dec 
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_19_fused_transpose_13"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_20_fused_reshape_13"
	//NO_APP
	dstcr	0x2, mode
.LBB0_141:                              // %loadstoreloop94
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_141
// %bb.142:                             // %split93
	dstcr	256, rp2
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 108, rp2
	dcp	rp3, r11
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	dcp	rp2, r10
	djal	:_Z16fused_reshape_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj169ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj507ELj1ELj85EEEEvRT0_RT1_
	daddi32	rp1, 108, rp2
	dstcr	0x2, mode
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	dcp	[rp2], link
	daddi32	rp1, 240, rp2
	dstcr	0, r10
	nop <> __iss__ print	489 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_20_fused_reshape_13"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_21_fused_transpose_12"
	//NO_APP
	dstcr	0x2, mode
.LBB0_143:                              // %loadstoreloop92
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_143
// %bb.144:                             // %split91
	daddi32	rp1, 104, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	dcp	rp3, r11
	daddi32	rp1, rp2, rp2
	dcp	rp2, r10
	djal	:_Z18fused_transpose_12I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj507ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj507EEEEvRT0_RT1_
	daddi32	rp1, 104, rp2
	dstcr	0x2, mode
	dstcr	9938752, r11
	dcp	[rp2], link
	daddi32	rp1, 244, rp2
	nop <> __iss__ print	501 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 244, rp2
	dshrab	r10, 31, r12
	daddi32	r19, r11, r10
	dandb	[rp2], r12, r11
	dcmplt32	r10, r19, r12
	daddi32	rp1, 240, rp2
	dcp	r11, dependencyid
	daddi32	r18, r12, r11
	dcp	[rp2], els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0xaa0, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0xaa0, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_145:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r12
	djmpneqoff	r12, 0, :.LBB0_145
// %bb.146:
	daddi32	rp1, 244, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	dstcr	260, rp3
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r13
	daddi32	rp2, 4, rp2
	dstcr	0, r12
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_21_fused_transpose_12"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_22_fused_split_1"
	//NO_APP
	dstcr	0x2, mode
.LBB0_147:                              // %loadstoreloop90
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r13, 1, r14
	dstcr	0, [rp2+=1]
	dcmplt32	r14, r13, r13
	dcmplt32	r14, 3, r15
	daddi32	r12, r13, r12
	dcp	r14, r13
	dcmpeq32	r12, 0, r14
	dcmpneq32	r14, 0, r14
	dcsel	r15, 0, r14
	djmpneqoff	r14, 0, :.LBB0_147
// %bb.148:                             // %split89
	daddi32	rp1, 208, rp2
	dstcr	256, rp3
	dstcr	0, r13
	daddi32	rp2, 4, rp2
	dstcr	0, r12
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_149:                              // %loadstoreloop88
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r13, 1, r14
	dstcr	0, [rp2+=1]
	dcmplt32	r14, r13, r13
	dcmplt32	r14, 3, r15
	daddi32	r12, r13, r12
	dcp	r14, r13
	dcmpeq32	r12, 0, r14
	dcmpneq32	r14, 0, r14
	dcsel	r15, 0, r14
	djmpneqoff	r14, 0, :.LBB0_149
// %bb.150:                             // %split87
	daddi32	rp1, 208, rp2
	dstcr	4198400, [rp2]
	dstcr	0x0, dependencyid
	dstcr	0x400000, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0x40, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0x40, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_151:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_151
// %bb.152:
	dstcr	8444992, r10
	dstcr	260, rp2
	daddi32	r19, r10, r10
	daddi32	rp1, rp2, rp2
	dcmplt32	r10, r19, r11
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x401000, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x400, els.intstride2
	dstcr	0x20, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x400, els.extstride2
	dstcr	0x20, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_153:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_153
// %bb.154:
	daddi32	rp1, 212, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	daddi32	rp1, 240, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_22_fused_split_1"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_23_fused_sigmoid_fixed_point_multiply_cast_cast_add"
	//NO_APP
	dstcr	0x2, mode
.LBB0_155:                              // %loadstoreloop86
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_155
// %bb.156:                             // %split85
	daddi32	rp1, 192, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_157:                              // %loadstoreloop84
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_157
// %bb.158:                             // %split83
	daddi32	rp1, 176, rp2
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, 192, rp3
	dstcr	4096, [rp3]
.LBB0_159:                              // %loadstoreloop82
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_159
// %bb.160:                             // %split81
	dstcr	9942848, r10
	daddi32	rp1, 176, rp2
	daddi32	r19, r10, r10
	dstcr	8192, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x1000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0x40, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0x40, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_161:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_161
// %bb.162:
	dstcr	8447040, r10
	daddi32	rp1, 196, rp2
	daddi32	r19, r10, r10
	dstcr	0x1, elsstatus
	dcmplt32	r10, r19, r11
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x2000, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x400, els.intstride2
	dstcr	0x20, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x400, els.extstride2
	dstcr	0x20, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_163:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_163
// %bb.164:
	daddi32	rp1, 180, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 100, rp2
	daddi32	rp1, 208, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	daddi32	rp1, 240, rp4
	daddi32	rp1, rp2, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z48fused_sigmoid_fixed_point_multiply_cast_cast_addI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 100, rp2
	dstcr	0x2, mode
	dstcr	10112832, r11
	dcp	[rp2], link
	daddi32	rp1, 244, rp2
	nop <> __iss__ print	549 -dec 
	dshlb	[rp2], 16, r10
	daddi32	r19, r11, r22
	dshrab	r10, 31, r10
	daddi32	rp1, 244, rp2
	dcmplt32	r22, r19, r11
	dandb	[rp2], r10, r10
	daddi32	rp1, 240, rp2
	dcp	r10, dependencyid
	daddi32	r18, r11, r23
	dcp	[rp2], els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0x40, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0x40, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_165:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_165
// %bb.166:
	daddi32	rp1, 244, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	dstcr	0, r11
	daddi32	rp1, rp3, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_23_fused_sigmoid_fixed_point_multiply_cast_cast_add"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_24_fused_exp_cast_multiply"
	//NO_APP
	dstcr	0x2, mode
.LBB0_167:                              // %loadstoreloop80
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_167
// %bb.168:                             // %split79
	daddi32	rp1, 208, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_169:                              // %loadstoreloop78
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_169
// %bb.170:                             // %split77
	dstcr	9946944, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	4198400, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x401000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0x20, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0x20, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_171:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_171
// %bb.172:
	daddi32	rp1, 212, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 96, rp2
	dstcr	256, rp4
	dcp	link, [rp2]
	daddi32	rp1, 192, rp2
	daddi32	rp1, 176, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z23fused_exp_cast_multiplyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 96, rp2
	dstcr	0x2, mode
	dstcr	10116928, r11
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	572 -dec 
	daddi32	rp1, rp2, rp2
	daddi32	r19, r11, r11
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dcmplt32	r11, r19, r12
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	daddi32	r18, r12, r10
	dcp	[rp2], els.intaddr
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0x40, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0x40, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_173:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_173
// %bb.174:
	dstcr	260, rp3
	dstcr	0x1, elsstatus
	daddi32	rp1, rp3, rp3
	dstcr	0, r11
	dcp	flowid, [rp3]
	daddi32	rp1, 180, rp3
	daddi32	rp1, 240, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 176, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 196, rp3
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	dstcr	0, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_24_fused_exp_cast_multiply"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_25_fused_sigmoid"
	//NO_APP
	dstcr	0x2, mode
.LBB0_175:                              // %loadstoreloop76
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_175
// %bb.176:                             // %split75
	daddi32	rp1, 92, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	daddi32	rp1, 208, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	djal	:_Z13fused_sigmoidI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj507EEES6_EvRT0_RT1_
	daddi32	rp1, 92, rp2
	dstcr	0x2, mode
	dstcr	10121024, r11
	dcp	[rp2], link
	daddi32	rp1, 244, rp2
	nop <> __iss__ print	590 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 244, rp2
	dshrab	r10, 31, r10
	daddi32	r19, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r19, r12
	daddi32	rp1, 240, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r18, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0x20, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0x20, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_177:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_177
// %bb.178:
	daddi32	rp1, 244, rp3
	daddi32	rp1, 208, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_25_fused_sigmoid"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_26_fused_concatenate_8"
	//NO_APP
	dstcr	0x2, mode
.LBB0_179:                              // %loadstoreloop74
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_179
// %bb.180:                             // %split73
	dstcr	9948992, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	4198400, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x401000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0xa00, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0xa00, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_181:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_181
// %bb.182:
	daddi32	rp1, 212, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	daddi32	rp1, 240, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_26_fused_concatenate_8"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_27_fused_sigmoid_1"
	//NO_APP
	dstcr	0x2, mode
.LBB0_183:                              // %loadstoreloop72
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_183
// %bb.184:                             // %split71
	daddi32	rp1, 88, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	daddi32	rp1, 208, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	djal	:_Z15fused_sigmoid_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj507EEES6_EvRT0_RT1_
	daddi32	rp1, 88, rp2
	dstcr	0x2, mode
	dstcr	10123072, r11
	dcp	[rp2], link
	daddi32	rp1, 244, rp2
	nop <> __iss__ print	620 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 244, rp2
	dshrab	r10, 31, r10
	daddi32	r19, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r19, r12
	daddi32	rp1, 240, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r18, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0xa00, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0xa00, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_185:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_185
// %bb.186:
	daddi32	rp1, 244, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 212, rp3
	daddi32	rp1, 208, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	//APP
	nop <> __iss__ profile stop -msg "Layer_27_fused_sigmoid_1"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_28_fused_concatenate_7"
	//NO_APP
	dstcr	0x2, mode
	dstcr	4198400, [rp3]
.LBB0_187:                              // %loadstoreloop70
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_187
// %bb.188:                             // %split69
	daddi32	rp1, 228, rp2
	dshlb	[rp2], 16, r10
	daddi32	rp1, 208, rp2
	dshrab	r10, 31, r10
	dstcr	4251648, [rp2]
	daddi32	rp1, 228, rp2
	dandb	[rp2], r10, r10
	dcp	r10, dependencyid
	dstcr	0x401000, els.intaddr
	dcp	r24, els.extaddrl
	dcp	r25, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x340, els.intstride2
	dstcr	0x340, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x340, els.extstride2
	dstcr	0x340, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_189:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_189
// %bb.190:
	dstcr	8449088, r10
	daddi32	rp1, 228, rp2
	daddi32	r19, r10, r10
	dstcr	0x1, elsstatus
	dcmplt32	r10, r19, r11
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x40e000, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x8800, els.intstride2
	dstcr	0x220, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x8800, els.extstride2
	dstcr	0x220, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_191:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_191
// %bb.192:
	daddi32	rp1, 212, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	daddi32	rp1, 240, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_28_fused_concatenate_7"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_29_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_16215914359837010491_"
	//NO_APP
	dstcr	0x2, mode
.LBB0_193:                              // %loadstoreloop68
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_193
// %bb.194:                             // %split67
	daddi32	rp1, 192, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_195:                              // %loadstoreloop66
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_195
// %bb.196:                             // %split65
	dstcr	8483904, r10
	daddi32	rp1, 192, rp2
	daddi32	r19, r10, r10
	dstcr	106496, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x1a000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0xc00, els.intstride2
	dstcr	0x30, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0xc00, els.extstride2
	dstcr	0x30, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_197:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_197
// %bb.198:
	daddi32	rp1, 196, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 84, rp2
	daddi32	rp1, 208, rp3
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	daddi32	rp1, 240, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z102fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_16215914359837010491_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj34816EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj128ELj13ELj13EEEEvRT0_RT1_RT2_
	daddi32	rp1, 84, rp2
	dstcr	0x2, mode
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	661 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 228, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_29_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_16215914359837010491_"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_30_fused_nn_conv2d_transpose_cast"
	//NO_APP
	dstcr	0x2, mode
.LBB0_199:                              // %loadstoreloop64
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_199
// %bb.200:                             // %split63
	dstcr	256, rp2
	dstcr	256, rp4
	daddi32	rp1, rp2, rp2
	daddi32	rp1, 192, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 80, rp2
	daddi32	rp1, rp4, rp4
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z30fused_nn_conv2d_transpose_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS1_L10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IS1_LS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj1536EEES2_IS0_IiLh16ELh4ELi0EELS3_0EjLj64ELS4_1EJLj1ELj128ELj26ELj26EEEEvRT0_RT1_RT2_
	daddi32	rp1, 80, rp2
	dstcr	0x2, mode
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	675 -dec 
	daddi32	rp1, rp2, rp2
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	dcp	[rp2], els.intaddr
	dcp	r20, els.extaddrl
	dcp	r21, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0xd00, els.intstride2
	dstcr	0x1a00, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0xd00, els.extstride2
	dstcr	0x1a00, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_201:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_201
// %bb.202:
	dstcr	260, rp3
	dstcr	0x1, elsstatus
	daddi32	rp1, rp3, rp3
	dstcr	0, r11
	dcp	flowid, [rp3]
	daddi32	rp1, 196, rp3
	daddi32	rp1, 240, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_30_fused_nn_conv2d_transpose_cast"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_31_fused_concatenate_13"
	//NO_APP
	dstcr	0x2, mode
.LBB0_203:                              // %loadstoreloop62
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_203
// %bb.204:                             // %split61
	dstcr	0x0, dependencyid
	dstcr	0x0, els.intaddr
	dcp	r20, els.extaddrl
	dcp	r21, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0xd00, els.intstride2
	dstcr	0x4e00, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0xd00, els.extstride2
	dstcr	0x4e00, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_205:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_205
// %bb.206:
	daddi32	rp1, 244, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	dstcr	260, rp3
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_31_fused_concatenate_13"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_32_fused_fixed_point_multiply_cast_round_clip_cast"
	//NO_APP
	dstcr	0x2, mode
.LBB0_207:                              // %loadstoreloop60
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_207
// %bb.208:                             // %split59
	daddi32	rp1, 224, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_209:                              // %loadstoreloop58
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_209
// %bb.210:                             // %split57
	dstcr	8486976, r10
	daddi32	rp1, 224, rp2
	daddi32	r19, r10, r10
	dstcr	4513792, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x44e000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0xd9000, els.intstride2
	dstcr	0x3640, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0xd9000, els.extstride2
	dstcr	0x3640, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_211:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_211
// %bb.212:
	daddi32	rp1, 228, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 76, rp2
	dstcr	256, rp3
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	daddi32	rp1, rp3, rp3
	dcp	rp2, r10
	dcp	rp3, r11
	djal	:_Z47fused_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj384ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj96ELj26ELj26EEEEvRT0_RT1_
	daddi32	rp1, 76, rp2
	dstcr	0x2, mode
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	dcp	[rp2], link
	daddi32	rp1, 240, rp2
	dstcr	0, r10
	nop <> __iss__ print	712 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_32_fused_fixed_point_multiply_cast_round_clip_cast"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_33_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__3"
	//NO_APP
	dstcr	0x2, mode
.LBB0_213:                              // %loadstoreloop56
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_213
// %bb.214:                             // %split55
	daddi32	rp1, 208, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_215:                              // %loadstoreloop54
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_215
// %bb.216:                             // %split53
	dstcr	9375808, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	212992, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x34000, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x10700, els.intstride2
	dstcr	0x41c, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x10700, els.extstride2
	dstcr	0x41c, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_217:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_217
// %bb.218:
	daddi32	rp1, 212, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 72, rp2
	daddi32	rp1, 224, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	daddi32	rp1, 240, rp4
	daddi32	rp1, rp2, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj96ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj888832EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj26ELj26EEEEvRT0_RT1_RT2_
	daddi32	rp1, 72, rp2
	dstcr	0x2, mode
	daddi32	rp1, 228, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	730 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_33_fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__3"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_34_fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943__1"
	//NO_APP
	dstcr	0x2, mode
.LBB0_219:                              // %loadstoreloop52
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_219
// %bb.220:                             // %split51
	dstcr	256, rp2
	dstcr	256, rp4
	daddi32	rp1, rp2, rp2
	daddi32	rp1, 208, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 68, rp2
	daddi32	rp1, rp4, rp4
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	dcp	rp3, r11
	dcp	rp2, r10
	dcp	rp4, r12
	djal	:_Z104fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj67320EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj26ELj26EEEEvRT0_RT1_RT2_
	daddi32	rp1, 68, rp2
	dstcr	0x2, mode
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	dcp	[rp2], link
	nop <> __iss__ print	743 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	daddi32	rp1, 240, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_34_fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943__1"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_35_fused_reshape_16"
	//NO_APP
	dstcr	0x2, mode
.LBB0_221:                              // %loadstoreloop50
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_221
// %bb.222:                             // %split49
	daddi32	rp1, 64, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	dcp	rp3, r11
	daddi32	rp1, rp2, rp2
	dcp	rp2, r10
	djal	:_Z16fused_reshape_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj26ELj26EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj676EEEEvRT0_RT1_
	daddi32	rp1, 64, rp2
	dstcr	0x2, mode
	dstcr	260, rp3
	dstcr	0, r11
	dcp	[rp2], link
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	nop <> __iss__ print	756 -dec 
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_35_fused_reshape_16"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_36_fused_transpose_16"
	//NO_APP
	dstcr	0x2, mode
.LBB0_223:                              // %loadstoreloop48
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_223
// %bb.224:                             // %split47
	dstcr	256, rp2
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 60, rp2
	dcp	rp3, r11
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	dcp	rp2, r10
	djal	:_Z18fused_transpose_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj676EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj676ELj1ELj255EEEEvRT0_RT1_
	daddi32	rp1, 60, rp2
	dstcr	0x2, mode
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	dcp	[rp2], link
	daddi32	rp1, 240, rp2
	dstcr	0, r10
	nop <> __iss__ print	768 -dec 
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_36_fused_transpose_16"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_37_fused_reshape_15"
	//NO_APP
	dstcr	0x2, mode
.LBB0_225:                              // %loadstoreloop46
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_225
// %bb.226:                             // %split45
	daddi32	rp1, 56, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	dcp	rp3, r11
	daddi32	rp1, rp2, rp2
	dcp	rp2, r10
	djal	:_Z16fused_reshape_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj676ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj2028ELj1ELj85EEEEvRT0_RT1_
	daddi32	rp1, 56, rp2
	dstcr	0x2, mode
	dstcr	260, rp3
	dstcr	0, r11
	dcp	[rp2], link
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	nop <> __iss__ print	780 -dec 
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_37_fused_reshape_15"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_38_fused_transpose_15"
	//NO_APP
	dstcr	0x2, mode
.LBB0_227:                              // %loadstoreloop44
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_227
// %bb.228:                             // %split43
	dstcr	256, rp2
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 52, rp2
	dcp	rp3, r11
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	dcp	rp2, r10
	djal	:_Z18fused_transpose_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2028ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj2028EEEEvRT0_RT1_
	daddi32	rp1, 52, rp2
	dstcr	0x2, mode
	dstcr	11564864, r11
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	792 -dec 
	daddi32	rp1, rp2, rp2
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r12
	daddi32	rp1, rp2, rp2
	daddi32	r19, r11, r10
	dandb	[rp2], r12, r11
	dstcr	256, rp2
	dcmplt32	r10, r19, r12
	daddi32	rp1, rp2, rp2
	dcp	r11, dependencyid
	daddi32	r18, r12, r11
	dcp	[rp2], els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0x2a2b, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0x2a2b, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_229:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r12
	djmpneqoff	r12, 0, :.LBB0_229
// %bb.230:
	dstcr	260, rp3
	dstcr	0x1, elsstatus
	daddi32	rp1, rp3, rp3
	dstcr	0, r13
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	daddi32	rp1, 240, rp2
	dstcr	0, r12
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_38_fused_transpose_15"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_39_fused_split_2"
	//NO_APP
	dstcr	0x2, mode
.LBB0_231:                              // %loadstoreloop42
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r13, 1, r14
	dstcr	0, [rp2+=1]
	dcmplt32	r14, r13, r13
	dcmplt32	r14, 4, r15
	daddi32	r12, r13, r12
	dcp	r14, r13
	dcmpeq32	r12, 0, r14
	dcmpneq32	r14, 0, r14
	dcsel	r15, 0, r14
	djmpneqoff	r14, 0, :.LBB0_231
// %bb.232:                             // %split41
	daddi32	rp1, 224, rp2
	dstcr	0, r12
	daddi32	rp2, 4, rp2
	dstcr	0, r13
.LBB0_233:                              // %loadstoreloop40
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r12, 1, r14
	dstcr	0, [rp2+=1]
	dcmplt32	r14, r12, r12
	dcmplt32	r14, 3, r15
	daddi32	r13, r12, r13
	dcp	r14, r12
	dcmpeq32	r13, 0, r14
	dcmpneq32	r14, 0, r14
	dcsel	r15, 0, r14
	djmpneqoff	r14, 0, :.LBB0_233
// %bb.234:                             // %split39
	daddi32	rp1, 224, rp2
	dstcr	16256, [rp2]
	dstcr	0x0, dependencyid
	dstcr	0x0, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0xfe, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0xfe, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_235:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_235
// %bb.236:
	dstcr	9443136, r10
	daddi32	rp1, 244, rp2
	daddi32	r19, r10, r10
	dstcr	0x1, elsstatus
	dcmplt32	r10, r19, r11
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x3f80, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1000, els.intstride2
	dstcr	0x80, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1000, els.extstride2
	dstcr	0x80, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_237:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_237
// %bb.238:
	daddi32	rp1, 228, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	dstcr	260, rp3
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_39_fused_split_2"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_40_fused_sigmoid_fixed_point_multiply_cast_cast_add_1"
	//NO_APP
	dstcr	0x2, mode
.LBB0_239:                              // %loadstoreloop38
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_239
// %bb.240:                             // %split37
	daddi32	rp1, 208, rp2
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp3]
.LBB0_241:                              // %loadstoreloop36
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_241
// %bb.242:                             // %split35
	daddi32	rp1, 192, rp2
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, 208, rp3
	dstcr	4210560, [rp3]
.LBB0_243:                              // %loadstoreloop34
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_243
// %bb.244:                             // %split33
	dstcr	11581120, r10
	daddi32	rp1, 192, rp2
	daddi32	r19, r10, r10
	dstcr	4226816, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x403f80, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0xfe, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0xfe, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_245:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_245
// %bb.246:
	dstcr	9451328, r10
	daddi32	rp1, 212, rp2
	daddi32	r19, r10, r10
	dstcr	0x1, elsstatus
	dcmplt32	r10, r19, r11
	dcp	flowid, [rp2]
	daddi32	r18, r11, r11
	dstcr	0x0, dependencyid
	dstcr	0x407f00, els.intaddr
	dcp	r10, els.extaddrl
	dcp	r11, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1000, els.intstride2
	dstcr	0x80, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1000, els.extstride2
	dstcr	0x80, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_247:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_247
// %bb.248:
	daddi32	rp1, 196, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 48, rp2
	dstcr	256, rp4
	dcp	link, [rp2]
	daddi32	rp1, 240, rp2
	daddi32	rp1, 224, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z50fused_sigmoid_fixed_point_multiply_cast_cast_add_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 48, rp2
	dstcr	0x2, mode
	dstcr	12255744, r11
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	840 -dec 
	daddi32	rp1, rp2, rp2
	daddi32	r19, r11, r20
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dcmplt32	r20, r19, r11
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	daddi32	r18, r11, r21
	dcp	[rp2], els.intaddr
	dcp	r20, els.extaddrl
	dcp	r21, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0xfe, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0xfe, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_249:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_249
// %bb.250:
	dstcr	260, rp3
	dstcr	0x1, elsstatus
	daddi32	rp1, rp3, rp3
	dstcr	0, r11
	dcp	flowid, [rp3]
	daddi32	rp1, 228, rp3
	daddi32	rp1, 240, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_40_fused_sigmoid_fixed_point_multiply_cast_cast_add_1"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_41_fused_exp_cast_multiply_1"
	//NO_APP
	dstcr	0x2, mode
.LBB0_251:                              // %loadstoreloop32
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_251
// %bb.252:                             // %split31
	daddi32	rp1, 224, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_253:                              // %loadstoreloop30
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_253
// %bb.254:                             // %split29
	dstcr	11597376, r10
	daddi32	rp1, 224, rp2
	daddi32	r19, r10, r10
	dstcr	16256, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x3f80, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0x7f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0x7f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_255:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_255
// %bb.256:
	daddi32	rp1, 228, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 44, rp2
	daddi32	rp1, 192, rp3
	dcp	link, [rp2]
	daddi32	rp1, 208, rp2
	daddi32	rp1, 240, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	djal	:_Z25fused_exp_cast_multiply_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_
	daddi32	rp1, 44, rp2
	dstcr	0x2, mode
	dstcr	12272000, r11
	dcp	[rp2], link
	daddi32	rp1, 244, rp2
	nop <> __iss__ print	863 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 244, rp2
	dshrab	r10, 31, r10
	daddi32	r19, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r19, r12
	daddi32	rp1, 240, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r18, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0xfe, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0xfe, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_257:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_257
// %bb.258:
	daddi32	rp1, 244, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 196, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	dstcr	260, rp3
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_41_fused_exp_cast_multiply_1"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_42_fused_sigmoid_2"
	//NO_APP
	dstcr	0x2, mode
.LBB0_259:                              // %loadstoreloop28
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_259
// %bb.260:                             // %split27
	dstcr	256, rp2
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 40, rp2
	dcp	rp3, r11
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	dcp	rp2, r10
	djal	:_Z15fused_sigmoid_2I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2028EEES6_EvRT0_RT1_
	daddi32	rp1, 40, rp2
	dstcr	0x2, mode
	dstcr	12288256, r11
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	881 -dec 
	daddi32	rp1, rp2, rp2
	daddi32	r19, r11, r11
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dcmplt32	r11, r19, r12
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	daddi32	r18, r12, r10
	dcp	[rp2], els.intaddr
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0x7f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0x7f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_261:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_261
// %bb.262:
	dstcr	260, rp3
	daddi32	rp1, 224, rp2
	daddi32	rp1, rp3, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 228, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_42_fused_sigmoid_2"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_43_fused_concatenate_12"
	//NO_APP
	dstcr	0x2, mode
.LBB0_263:                              // %loadstoreloop26
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_263
// %bb.264:                             // %split25
	dstcr	11605504, r10
	daddi32	rp1, 224, rp2
	daddi32	r19, r10, r10
	dstcr	16256, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x3f80, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0x27b0, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0x27b0, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_265:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_265
// %bb.266:
	daddi32	rp1, 228, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	dstcr	260, rp3
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	dstcr	256, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_43_fused_concatenate_12"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_44_fused_sigmoid_3"
	//NO_APP
	dstcr	0x2, mode
.LBB0_267:                              // %loadstoreloop24
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_267
// %bb.268:                             // %split23
	dstcr	256, rp2
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	dstcr	4194304, [rp2]
	daddi32	rp1, 36, rp2
	dcp	rp3, r11
	dcp	link, [rp2]
	daddi32	rp1, 224, rp2
	dcp	rp2, r10
	djal	:_Z15fused_sigmoid_3I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj2028EEES6_EvRT0_RT1_
	daddi32	rp1, 36, rp2
	dstcr	0x2, mode
	dstcr	12296384, r11
	dcp	[rp2], link
	dstcr	260, rp2
	nop <> __iss__ print	911 -dec 
	daddi32	rp1, rp2, rp2
	daddi32	r19, r11, r11
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dcmplt32	r11, r19, r12
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	daddi32	r18, r12, r10
	dcp	[rp2], els.intaddr
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0x27b0, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0x27b0, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_269:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_269
// %bb.270:
	dstcr	260, rp2
	dstcr	0x1, elsstatus
	daddi32	rp1, rp2, rp2
	dcp	flowid, [rp2]
	daddi32	rp1, 228, rp2
	dstcr	0, [rp2]
	daddi32	rp1, 224, rp2
	dstcr	0, [rp2]
	//APP
	nop <> __iss__ profile stop -msg "Layer_44_fused_sigmoid_3"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_45_fused_concatenate_11"
	//NO_APP
	dstcr	0x0, dependencyid
	dstcr	0x3f80, els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x800, els.intstride2
	dstcr	0xaa0, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x800, els.extstride2
	dstcr	0xaa0, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_271:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_271
// %bb.272:
	dstcr	0x1, elsstatus
	dcp	flowid, r28
	dstcr	0x0, dependencyid
	dstcr	0x2e780, els.intaddr
	dcp	r20, els.extaddrl
	dcp	r21, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x1fc0, els.intstride2
	dstcr	0x2a2b, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x1fc0, els.extstride2
	dstcr	0x2a2b, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_273:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_273
// %bb.274:
	dstcr	260, rp2
	dstcr	0x1, elsstatus
	daddi32	rp1, rp2, rp2
	dcp	flowid, r7
	dstcr	0x2, mode
	dstcr	0, r29
	dstcr	16256, r10
	dstcr	0, [rp2]
	dstcr	256, rp2
	dstcr	1024, r11
	daddi32	rp1, rp2, rp2
	dstcr	10176, r12
	dstcr	4194304, r13
	addi32	crp1, 8, crp2           //      
	dstcr	8128, r14
	dstcr	190336, r15
	dstcr	7168, r16
	dstcr	2544, r17
	dstcr	507, r5
	stcr	2028, cr10
	dstcr	0, r6
	dstcr	0, [rp2]
	//APP
	nop <> __iss__ profile stop -msg "Layer_45_fused_concatenate_11"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_46_fused_concatenate_6"
	//NO_APP
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x9f0, pls.stride2, north
.LBB0_276:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_277 Depth 2
                                        //     Child Loop BB0_279 Depth 2
                                        //     Child Loop BB0_281 Depth 2
                                        //     Child Loop BB0_283 Depth 2
                                        //     Child Loop BB0_287 Depth 2
                                        //     Child Loop BB0_289 Depth 2
                                        //     Child Loop BB0_293 Depth 2
                                        //     Child Loop BB0_295 Depth 2
                                        //     Child Loop BB0_297 Depth 2
                                        //       Child Loop BB0_298 Depth 3
                                        //       Child Loop BB0_300 Depth 3
                                        //     Child Loop BB0_303 Depth 2
                                        //     Child Loop BB0_305 Depth 2
                                        //     Child Loop BB0_309 Depth 2
                                        //       Child Loop BB0_311 Depth 3
                                        //       Child Loop BB0_313 Depth 3
                                        //       Child Loop BB0_315 Depth 3
                                        //       Child Loop BB0_317 Depth 3
	dshlb	r6, 11, r30
	dstcr	0x11, pls.mode, south
	daddi32	r30, r10, r30
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr11
	cp	col, cr12
	dstcr	0x0, pc.constant, south
	dcp	r30, pls.addr, south
	dstcr	0x10, pls.count1, south
	dstcr	0x200, pls.stride2, south
	dcp	r28, dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, r28
	djmpincsetup	0, 16, :.LBB0_277
	dstcr	0x300, pc.mode, south
.LBB0_277:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.278:                             //   in Loop: Header=BB0_276 Depth=1
	djmpincsetup	0, 4, :.LBB0_279
	dstcr	0x200, pc.mode, south
.LBB0_279:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.280:                             //   in Loop: Header=BB0_276 Depth=1
	daddi32	r30, r11, r30
	cp	south, cr13
	dcp	r30, pls.addr, south
	dstcr	0x10, pls.count1, south
	dstcr	0x200, pls.stride2, south
	dcp	r28, dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, r28
	djmpincsetup	0, 16, :.LBB0_281
	dstcr	0x300, pc.mode, south
.LBB0_281:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.282:                             //   in Loop: Header=BB0_276 Depth=1
	djmpincsetup	0, 4, :.LBB0_283
	dstcr	0x200, pc.mode, south
.LBB0_283:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.284:                             //   in Loop: Header=BB0_276 Depth=1
	addi32	cr11, cr12, cr12
	stcr	0, cr11
	cmplti32	cr12, 251, cr12
	dstcr	0x200, pc.mode, south
	predpush	cr12, :.LBB0_286
// %bb.285:                             //   in Loop: Header=BB0_276 Depth=1
	cp	south, cr11
.LBB0_286:                              //   in Loop: Header=BB0_276 Depth=1
	predpop	
	dmuli32	r6, r12, r30
	dstcr	0x200, pc.mode, south
	shlb	row, 4, cr12
	daddi32	r30, r13, r30
	cp	col, cr14
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r30, pls.addr, north
	dcp	r29, dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, r29
	djmpincsetup	0, 4, :.LBB0_287
	nrb	cr13, north
	dstcr	0x200, pc.mode, north
.LBB0_287:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.288:                             //   in Loop: Header=BB0_276 Depth=1
	djmpincsetup	0, 16, :.LBB0_289
	dstcr	0x300, pc.mode, north
.LBB0_289:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.290:                             //   in Loop: Header=BB0_276 Depth=1
	daddi32	r30, r11, r30
	addi32	cr12, cr14, cr12
	dcp	r30, pls.addr, north
	dcp	r29, dependencyid
	dstcr	0x1, plsstatus, north
	cmplti32	cr12, 251, cr12
	dcp	flowid, r29
	predpush	cr12, :.LBB0_292
// %bb.291:                             //   in Loop: Header=BB0_276 Depth=1
	nrb	cr11, north
.LBB0_292:                              //   in Loop: Header=BB0_276 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB0_293
	dstcr	0x200, pc.mode, north
.LBB0_293:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.294:                             //   in Loop: Header=BB0_276 Depth=1
	djmpincsetup	0, 16, :.LBB0_295
	dstcr	0x300, pc.mode, north
.LBB0_295:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.296:                             //   in Loop: Header=BB0_276 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	0x400000, pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	cp	row, cr12
	cp	col, cr11
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dstcr	0, r30
	cp	crp2, crp3
	shlb	row, 4, cr13
	cp	col, cr14
	dstcr	0x0, pc.constant, south
	dstcr	0x10, pls.count1, south
	dstcr	0x7f0, pls.stride2, south
.LBB0_297:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_298 Depth 3
                                        //       Child Loop BB0_300 Depth 3
	dmuli32	r6, r14, r2
	dshlb	r30, 10, r20
	djmpincsetup	0, 16, :.LBB0_298
	daddi32	r2, r15, r2
	daddi32	r2, r20, r20
	dcp	r20, pls.addr, south
	dcp	r7, dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, r7
	dstcr	0x300, pc.mode, south
.LBB0_298:                              //   Parent Loop BB0_276 Depth=1
                                        //     Parent Loop BB0_297 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.299:                             //   in Loop: Header=BB0_297 Depth=2
	djmpincsetup	0, 4, :.LBB0_300
	dstcr	0x200, pc.mode, south
.LBB0_300:                              //   Parent Loop BB0_276 Depth=1
                                        //     Parent Loop BB0_297 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.301:                             //   in Loop: Header=BB0_297 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r30, 7, :.LBB0_297
// %bb.302:                             //   in Loop: Header=BB0_276 Depth=1
	daddi32	r2, r16, r2
	dmuli32	r6, r17, r30
	dcp	r2, pls.addr, south
	dstcr	0xf, pls.count1, south
	dstcr	0x7f0, pls.stride2, south
	dcp	r7, dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, r7
	djmpincsetup	0, 15, :.LBB0_303
	dstcr	0x300, pc.mode, south
.LBB0_303:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.304:                             //   in Loop: Header=BB0_276 Depth=1
	addi32	cr13, cr14, cr13
	djmpincsetup	0, 4, :.LBB0_305
	dstcr	0x200, pc.mode, south
.LBB0_305:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.306:                             //   in Loop: Header=BB0_276 Depth=1
	daddi32	r30, r5, r30
	cmplti32	cr13, 236, cr14
	stcr	0, cr13
	dstcr	0x200, pc.mode, south
	nnb	south, north
	predpush	cr14, :.LBB0_308
// %bb.307:                             //   in Loop: Header=BB0_276 Depth=1
	cp	south, cr13
.LBB0_308:                              //   in Loop: Header=BB0_276 Depth=1
	predpop	
	addi32	crp2, 28, crp3
	shlb	cr12, 4, cr12
	cp	cr13, [crp3]
	addi32	cr11, cr12, cr11
	dstcr	0, r2
	cp	crp2, crp3
	dstcr	0x200, pc.mode, south
.LBB0_309:                              //   Parent Loop BB0_276 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_311 Depth 3
                                        //       Child Loop BB0_313 Depth 3
                                        //       Child Loop BB0_315 Depth 3
                                        //       Child Loop BB0_317 Depth 3
	dshlb	r2, 8, r20
	dcpc	r20, cr12
	addi32	cr11, cr12, cr12
	cmplt32	cr12, cr10, cr13
	predpush	cr13, :.LBB0_319
// %bb.310:                             //   in Loop: Header=BB0_309 Depth=2
	dcpc	r30, cr13
	addi32	cr13, cr12, cr12
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	r29, dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, r29
	shlb	cr12, 2, cr12
	djmpincsetup	0, 4, :.LBB0_311
	dstcr	0x260, pc.mode, north
	nrb	cr12, north
.LBB0_311:                              //   Parent Loop BB0_276 Depth=1
                                        //     Parent Loop BB0_309 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.312:                             //   in Loop: Header=BB0_309 Depth=2
	djmpincsetup	0, 16, :.LBB0_313
	dstcr	0x360, pc.mode, north
.LBB0_313:                              //   Parent Loop BB0_276 Depth=1
                                        //     Parent Loop BB0_309 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.314:                             //   in Loop: Header=BB0_309 Depth=2
	dstcr	0x260, pc.mode, north
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB0_315
.LBB0_315:                              //   Parent Loop BB0_276 Depth=1
                                        //     Parent Loop BB0_309 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.316:                             //   in Loop: Header=BB0_309 Depth=2
	djmpincsetup	0, 16, :.LBB0_317
	dstcr	0x360, pc.mode, north
.LBB0_317:                              //   Parent Loop BB0_276 Depth=1
                                        //     Parent Loop BB0_309 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.318:                             //   in Loop: Header=BB0_309 Depth=2
	dstcr	0x260, pc.mode, north
.LBB0_319:                              // %Flow
                                        //   in Loop: Header=BB0_309 Depth=2
	predpop	
	addi32	crp3, 4, crp3
	djmpincne	r2, 8, :.LBB0_309
// %bb.275:                             //   in Loop: Header=BB0_276 Depth=1
	djmpincne	r6, 85, :.LBB0_276
// %bb.320:
	dshlb	r29, 16, r10
	dstcr	12946624, r11
	dshrab	r10, 31, r10
	daddi32	r19, r11, r20
	dandb	r10, r29, r10
	nop <> __iss__ print	947 -dec 
	dcp	r10, dependencyid
	dcmplt32	r20, r19, r10
	dstcr	0x400000, els.intaddr
	daddi32	r18, r10, r21
	dcp	r20, els.extaddrl
	dcp	r21, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x34cb, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x34cb, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_321:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_321
// %bb.322:
	dstcr	256, rp2
	daddi32	rp1, 244, rp3
	dstcr	0, r11
	daddi32	rp1, rp2, rp2
	dstcr	0, r10
	dstcr	0x1, elsstatus
	dstcr	0x2, mode
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_46_fused_concatenate_6"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_47_fused_split"
	//NO_APP
	dstcr	0x2, mode
.LBB0_323:                              // %loadstoreloop22
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_323
// %bb.324:                             // %split21
	dstcr	12966976, r10
	dstcr	0x0, dependencyid
	daddi32	r19, r10, r10
	dstcr	0x0, els.intaddr
	dcp	r10, els.extaddrl
	dcmplt32	r10, r19, r10
	daddi32	r18, r10, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_325:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_325
// %bb.326:
	dstcr	260, rp3
	daddi32	rp1, 240, rp2
	daddi32	rp1, rp3, rp3
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_47_fused_split"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_48_fused_fixed_point_multiply_cast"
	//NO_APP
	dstcr	0x2, mode
.LBB0_327:                              // %loadstoreloop20
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_327
// %bb.328:                             // %split19
	daddi32	rp1, 224, rp2
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, 240, rp3
	dstcr	4194304, [rp3]
.LBB0_329:                              // %loadstoreloop18
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_329
// %bb.330:                             // %split17
	daddi32	rp1, 224, rp2
	dstcr	4204480, [rp2]
	dstcr	0x0, dependencyid
	dstcr	0x4027c0, els.intaddr
	dcp	r20, els.extaddrl
	dcp	r21, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_331:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_331
// %bb.332:
	daddi32	rp1, 228, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 32, rp2
	daddi32	rp1, 240, rp3
	dcp	link, [rp2]
	dstcr	256, rp2
	dcp	rp3, r11
	daddi32	rp1, rp2, rp2
	dcp	rp2, r10
	djal	:_Z31fused_fixed_point_multiply_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_EvRT0_RT1_
	daddi32	rp1, 32, rp2
	dstcr	0x2, mode
	dstcr	13811584, r11
	dcp	[rp2], link
	daddi32	rp1, 244, rp2
	nop <> __iss__ print	985 -dec 
	dshlb	[rp2], 16, r10
	daddi32	r19, r11, r22
	dshrab	r10, 31, r10
	daddi32	rp1, 244, rp2
	dcmplt32	r22, r19, r11
	dandb	[rp2], r10, r10
	daddi32	rp1, 240, rp2
	dcp	r10, dependencyid
	daddi32	r18, r11, r23
	dcp	[rp2], els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_333:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_333
// %bb.334:
	daddi32	rp1, 244, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	dstcr	260, rp3
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_48_fused_fixed_point_multiply_cast"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_49_fused_subtract"
	//NO_APP
	dstcr	0x2, mode
.LBB0_335:                              // %loadstoreloop16
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_335
// %bb.336:                             // %split15
	daddi32	rp1, 208, rp2
	dstcr	0, r10
	daddi32	rp2, 4, rp2
	dstcr	0, r11
.LBB0_337:                              // %loadstoreloop14
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r10, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r10, r10
	dcmplt32	r12, 3, r13
	daddi32	r11, r10, r11
	dcp	r12, r10
	dcmpeq32	r11, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_337
// %bb.338:                             // %split13
	dstcr	12977152, r10
	daddi32	rp1, 208, rp2
	daddi32	r19, r10, r10
	dstcr	10176, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x27c0, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_339:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_339
// %bb.340:
	daddi32	rp1, 212, rp2
	dstcr	256, rp4
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 224, rp2
	daddi32	rp1, 240, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	dcp	link, [rp1 + 7]
	djal	:_Z14fused_subtractI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
	dstcr	260, rp2
	dstcr	0x2, mode
	daddi32	rp1, rp2, rp2
	dcp	[rp1 + 7], link
	nop <> __iss__ print	1007 -dec 
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcp	r10, dependencyid
	daddi32	rp1, rp2, rp2
	dcp	[rp2], els.intaddr
	dcp	r9, els.extaddrl
	dcp	r8, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_341:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_341
// %bb.342:
	dstcr	260, rp3
	dstcr	0x1, elsstatus
	daddi32	rp1, rp3, rp3
	daddi32	rp1, 192, rp2
	dcp	flowid, [rp3]
	daddi32	rp1, 228, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_49_fused_subtract"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_50_fused_fixed_point_multiply_cast"
	//NO_APP
	dstcr	0x2, mode
.LBB0_343:                              // %loadstoreloop12
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_343
// %bb.344:                             // %split11
	daddi32	rp1, 176, rp2
	dstcr	0, r11
	daddi32	rp2, 4, rp2
	dstcr	0, r10
	daddi32	rp1, 192, rp3
	dstcr	4194304, [rp3]
.LBB0_345:                              // %loadstoreloop10
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_345
// %bb.346:                             // %split9
	dstcr	12956800, r10
	daddi32	rp1, 176, rp2
	daddi32	r19, r10, r24
	dstcr	4204480, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r24, r19, r10
	dstcr	0x4027c0, els.intaddr
	daddi32	r18, r10, r25
	dcp	r24, els.extaddrl
	dcp	r25, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_347:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_347
// %bb.348:
	daddi32	rp1, 180, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 208, rp2
	daddi32	rp1, 192, rp3
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	link, [rp1 + 6]
	djal	:_Z31fused_fixed_point_multiply_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_EvRT0_RT1_
	dstcr	0x2, mode
	daddi32	rp1, 196, rp2
	dstcr	13831936, r11
	dcp	[rp1 + 6], link
	nop <> __iss__ print	1029 -dec 
	dshlb	[rp2], 16, r10
	daddi32	r19, r11, r26
	dshrab	r10, 31, r10
	daddi32	rp1, 196, rp2
	dcmplt32	r26, r19, r11
	dandb	[rp2], r10, r10
	daddi32	rp1, 192, rp2
	dcp	r10, dependencyid
	daddi32	r18, r11, r27
	dcp	[rp2], els.intaddr
	dcp	r26, els.extaddrl
	dcp	r27, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_349:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_349
// %bb.350:
	daddi32	rp1, 196, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 212, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 208, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	dstcr	260, rp3
	daddi32	rp1, rp2, rp2
	daddi32	rp1, rp3, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_50_fused_fixed_point_multiply_cast"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_51_fused_subtract"
	//NO_APP
	dstcr	0x2, mode
.LBB0_351:                              // %loadstoreloop8
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_351
// %bb.352:                             // %split7
	daddi32	rp1, 240, rp2
	dstcr	20352, [rp2]
	daddi32	rp1, 224, rp2
	dstcr	10176, [rp2]
	daddi32	rp1, 228, rp2
	dshlb	[rp2], 16, r10
	daddi32	rp1, 228, rp2
	dshrab	r10, 31, r10
	dandb	[rp2], r10, r10
	dcp	r10, dependencyid
	dstcr	0x27c0, els.intaddr
	dcp	r20, els.extaddrl
	dcp	r21, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_353:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_353
// %bb.354:
	daddi32	rp1, 228, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 244, rp2
	dshlb	[rp2], 16, r10
	daddi32	rp1, 244, rp2
	dshrab	r10, 31, r10
	dandb	[rp2], r10, r10
	dcp	r10, dependencyid
	dstcr	0x4f80, els.intaddr
	dcp	r22, els.extaddrl
	dcp	r23, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_355:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_355
// %bb.356:
	daddi32	rp1, 244, rp2
	dstcr	256, rp4
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 176, rp2
	daddi32	rp1, 192, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	dcp	link, [rp1 + 5]
	djal	:_Z14fused_subtractI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
	dstcr	260, rp2
	dstcr	0x2, mode
	daddi32	rp1, rp2, rp2
	dstcr	10176, r11
	dcp	[rp1 + 5], link
	nop <> __iss__ print	1056 -dec 
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	daddi32	r9, r11, r11
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcmplt32	r11, r9, r12
	daddi32	rp1, rp2, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r8, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_357:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_357
// %bb.358:
	dstcr	260, rp3
	dstcr	0x1, elsstatus
	daddi32	rp1, rp3, rp3
	daddi32	rp1, 208, rp2
	dcp	flowid, [rp3]
	daddi32	rp1, 180, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 176, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 196, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_51_fused_subtract"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_52_fused_add_14"
	//NO_APP
	dstcr	0x2, mode
.LBB0_359:                              // %loadstoreloop6
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_359
// %bb.360:                             // %split5
	daddi32	rp1, 192, rp2
	dstcr	4214656, [rp2]
	daddi32	rp1, 176, rp2
	dstcr	4204480, [rp2]
	daddi32	rp1, 180, rp2
	dshlb	[rp2], 16, r10
	daddi32	rp1, 208, rp2
	dshrab	r10, 31, r10
	dstcr	4194304, [rp2]
	daddi32	rp1, 180, rp2
	dandb	[rp2], r10, r10
	dcp	r10, dependencyid
	dstcr	0x4027c0, els.intaddr
	dcp	r24, els.extaddrl
	dcp	r25, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_361:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_361
// %bb.362:
	daddi32	rp1, 180, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 196, rp2
	dshlb	[rp2], 16, r10
	daddi32	rp1, 196, rp2
	dshrab	r10, 31, r10
	dandb	[rp2], r10, r10
	dcp	r10, dependencyid
	dstcr	0x404f80, els.intaddr
	dcp	r26, els.extaddrl
	dcp	r27, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_363:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_363
// %bb.364:
	daddi32	rp1, 196, rp2
	dstcr	0x1, elsstatus
	dcp	flowid, [rp2]
	daddi32	rp1, 224, rp2
	daddi32	rp1, 240, rp3
	daddi32	rp1, 208, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	dcp	link, [rp1 + 4]
	djal	:_Z12fused_add_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
	dstcr	0x2, mode
	daddi32	rp1, 212, rp2
	dstcr	20352, r11
	dcp	[rp1 + 4], link
	nop <> __iss__ print	1082 -dec 
	dshlb	[rp2], 16, r10
	daddi32	rp1, 212, rp2
	dshrab	r10, 31, r10
	daddi32	r9, r11, r11
	dandb	[rp2], r10, r10
	dcmplt32	r11, r9, r12
	daddi32	rp1, 208, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r8, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_365:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_365
// %bb.366:
	daddi32	rp1, 212, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	daddi32	rp1, 244, rp3
	dstcr	256, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 240, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 228, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 224, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	dstcr	260, rp3
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_52_fused_add_14"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_53_fused_add_14"
	//NO_APP
	dstcr	0x2, mode
.LBB0_367:                              // %loadstoreloop4
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_367
// %bb.368:                             // %split3
	dstcr	256, rp4
	daddi32	rp1, 176, rp2
	daddi32	rp1, 192, rp3
	daddi32	rp1, rp4, rp4
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	rp4, r12
	dcp	link, [rp1 + 3]
	djal	:_Z12fused_add_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
	dstcr	260, rp2
	dstcr	0x2, mode
	daddi32	rp1, rp2, rp2
	dstcr	30528, r11
	dcp	[rp1 + 3], link
	nop <> __iss__ print	1099 -dec 
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	daddi32	r9, r11, r11
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcmplt32	r11, r9, r12
	daddi32	rp1, rp2, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r8, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x9f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x9f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_369:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_369
// %bb.370:
	dstcr	260, rp3
	dstcr	0x1, elsstatus
	daddi32	rp1, rp3, rp3
	daddi32	rp1, 240, rp2
	dcp	flowid, [rp3]
	daddi32	rp1, 196, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	daddi32	rp1, 192, rp3
	daddi32	rp2, 4, rp2
	dstcr	0, [rp3]
	daddi32	rp1, 180, rp3
	dstcr	0, r10
	dstcr	0, [rp3]
	daddi32	rp1, 176, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_53_fused_add_14"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_54_fused_concatenate_5"
	//NO_APP
	dstcr	0x2, mode
.LBB0_371:                              // %loadstoreloop2
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 3, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_371
// %bb.372:                             // %split1
	dstcr	12987328, r10
	daddi32	rp1, 240, rp2
	daddi32	r19, r10, r10
	dstcr	4204480, [rp2]
	dstcr	0x0, dependencyid
	dcmplt32	r10, r19, r11
	dstcr	0x4027c0, els.intaddr
	dcp	r10, els.extaddrl
	daddi32	r18, r11, r10
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x324f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x324f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
.LBB0_373:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_373
// %bb.374:
	daddi32	rp1, 244, rp3
	dstcr	0x1, elsstatus
	dcp	flowid, [rp3]
	dstcr	260, rp3
	dstcr	256, rp2
	daddi32	rp1, rp3, rp3
	dstcr	0, r11
	dstcr	0, [rp3]
	dstcr	256, rp3
	daddi32	rp1, rp2, rp2
	dstcr	0, r10
	daddi32	rp1, rp3, rp3
	dstcr	0, [rp3]
	//APP
	nop <> __iss__ profile stop -msg "Layer_54_fused_concatenate_5"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "Layer_55_fused_copy"
	//NO_APP
	dstcr	0x2, mode
.LBB0_375:                              // %loadstoreloop
                                        // =>This Inner Loop Header: Depth=1
	daddi32	r11, 1, r12
	dstcr	0, [rp2+=1]
	dcmplt32	r12, r11, r11
	dcmplt32	r12, 4, r13
	daddi32	r10, r11, r10
	dcp	r12, r11
	dcmpeq32	r10, 0, r12
	dcmpneq32	r12, 0, r12
	dcsel	r13, 0, r12
	djmpneqoff	r12, 0, :.LBB0_375
// %bb.376:                             // %split
	dstcr	256, rp3
	daddi32	rp1, 240, rp2
	daddi32	rp1, rp3, rp3
	dcp	rp2, r10
	dcp	rp3, r11
	dcp	link, [rp1 + 2]
	djal	:_Z10fused_copyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj81ELj1ELj2535EEES6_EvRT0_RT1_
	dstcr	260, rp2
	dstcr	0x2, mode
	daddi32	rp1, rp2, rp2
	dstcr	40704, r11
	dcp	[rp1 + 2], link
	nop <> __iss__ print	1129 -dec 
	dshlb	[rp2], 16, r10
	dstcr	260, rp2
	dshrab	r10, 31, r10
	daddi32	rp1, rp2, rp2
	daddi32	r9, r11, r11
	dandb	[rp2], r10, r10
	dstcr	256, rp2
	dcmplt32	r11, r9, r12
	daddi32	rp1, rp2, rp2
	dcp	r10, dependencyid
	dcp	[rp2], els.intaddr
	daddi32	r8, r12, r10
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x27c0, els.intstride2
	dstcr	0x324f, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x27c0, els.extstride2
	dstcr	0x324f, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
.LBB0_377:                              // =>This Inner Loop Header: Depth=1
	dandb	elsstatus, 2, r10
	djmpneqoff	r10, 0, :.LBB0_377
// %bb.378:
	dstcr	260, rp2
	dstcr	0x1, elsstatus
	daddi32	rp1, rp2, rp2
	dcp	flowid, [rp2]
	daddi32	rp1, 244, rp2
	dstcr	0, [rp2]
	daddi32	rp1, 240, rp2
	dstcr	0, [rp2]
	dstcr	260, rp2
	//APP
	nop <> __iss__ profile stop -msg "Layer_55_fused_copy"
	//NO_APP
	daddi32	rp1, rp2, rp2
	//APP
	nop <> __iss__ profile start -msg "Layer_56_fused_concatenate_4"
	//NO_APP
	dstcr	0x2, mode
	dstcr	0, [rp2]
	dstcr	256, rp2
	daddi32	rp1, rp2, rp2
	dstcr	0, [rp2]
	daddi32	rp1, 212, rp2
	dstcr	0, [rp2]
	daddi32	rp1, 208, rp2
	dstcr	0, [rp2]
	//APP
	nop <> __iss__ profile stop -msg "Layer_56_fused_concatenate_4"
	//NO_APP
	dendk	
                                        // -- End function
	.p2align	3               // -- Begin function _Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__4I10FixedPointIsLh6ELh2ELi0EE7_TensorIaL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj3ELj416ELj416EEES2_IaLS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj896EEES2_IDv4_aLS3_0EjLj64ELS4_1EJLj1ELj4ELj208ELj208EEEEvRT0_RT1_RT2_
_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__4I10FixedPointIsLh6ELh2ELi0EE7_TensorIaL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj3ELj416ELj416EEES2_IaLS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj896EEES2_IDv4_aLS3_0EjLj64ELS4_1EJLj1ELj4ELj208ELj208EEEEvRT0_RT1_RT2_: // @_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__4I10FixedPointIsLh6ELh2ELi0EE7_TensorIaL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj3ELj416ELj416EEES2_IaLS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj896EEES2_IDv4_aLS3_0EjLj64ELS4_1EJLj1ELj4ELj208ELj208EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -96, rp1
	daddi32	rp1, 88, rp2
	dstcr	0x2, mode
	addi32	crp1, -24, crp1         //     
	dcp	r11, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 80, rp2
	dcp	r10, rp4
	dcp	r9, [rp2]
	daddi32	rp1, 72, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 64, rp2
	dstcr	65535, r11
	dcp	r19, [rp2]
	daddi32	rp1, 56, rp2
	dstcr	448, r13
	dcp	r20, [rp2]
	daddi32	rp1, 48, rp2
	dstcr	65520, r14
	dcp	r21, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	416, r15
	dcp	r22, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	-1, r16
	dcp	r23, [rp2]
	dcp	r12, rp2
	dstcr	1321528399, r12
	dstcr	400, r17
	dstcr	650, r5
	dstcr	-676, r6
	dstcr	649, r7
	addi32	crp1, 16, crp2          //      
	stcr	11135, cr10
	stcr	12195, cr11
	stcr	12885, cr12
	stcr	6553, cr13
	stcr	15146, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	65528, r28
	dstcr	0, rp5
	dstcr	43264, r29
	dstcr	676, r30
	dcp	r24, [rp1 + 6]
	dcp	r25, [rp1 + 4]
	dcp	r26, [rp1 + 2]
	dcp	r27, [rp1]
	dstcr	0x1c0, pls.stride1, south
	dstcr	0x3, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x2d800, pls.stride2, south
	dstcr	0x0, plsthresholdnorth
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x70, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB1_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB1_5 Depth 2
                                        //       Child Loop BB1_6 Depth 3
                                        //       Child Loop BB1_8 Depth 3
                                        //       Child Loop BB1_10 Depth 3
                                        //     Child Loop BB1_15 Depth 2
                                        //       Child Loop BB1_16 Depth 3
                                        //       Child Loop BB1_18 Depth 3
                                        //     Child Loop BB1_24 Depth 2
                                        //       Child Loop BB1_25 Depth 3
                                        //       Child Loop BB1_27 Depth 3
                                        //     Child Loop BB1_32 Depth 2
                                        //       Child Loop BB1_33 Depth 3
                                        //     Child Loop BB1_40 Depth 2
                                        //       Child Loop BB1_41 Depth 3
                                        //       Child Loop BB1_43 Depth 3
                                        //     Child Loop BB1_48 Depth 2
                                        //       Child Loop BB1_49 Depth 3
                                        //     Child Loop BB1_55 Depth 2
                                        //       Child Loop BB1_56 Depth 3
                                        //     Child Loop BB1_61 Depth 2
                                        //     Child Loop BB1_63 Depth 2
                                        //     Child Loop BB1_65 Depth 2
                                        //       Child Loop BB1_68 Depth 3
                                        //       Child Loop BB1_70 Depth 3
                                        //       Child Loop BB1_72 Depth 3
                                        //       Child Loop BB1_74 Depth 3
	dandb	r10, r11, r9
	daddi32	r10, r6, r2
	dcmplt32	r9, 26, r8
	dmul32hi	r9, r12, r8
	dandb	r2, r11, r18
	dstcr	1, r20
	dshrlb	r8, 3, r2
	dmuli32	r2, -26, r8
	dshlb	r2, 4, r19
	daddi32	r8, r10, r8
	dcsel	r19, 4, r21
	dshlb	r8, 4, r24
	dsubi32	r19, r21, r23
	dandb	r24, r14, r24
	dsubi32	r17, r19, r19
	dsubi32	r15, r24, r24
	dmaxi32	r19, 0, r25
	dmini32	r24, 20, r19
	dcmplt32	25, r9, r22
	dcsel	r23, 0, r23
	dsubi32	60, r19, r26
	dcmplti32	28, r19, r27
	dsubi32	28, r19, r27
	dshrlb	r16, r26, r26
	dshrlb	r16, r27, r27
	dmuli32	r23, r13, r23
	dandb	r8, r11, r24
	dcsel	r26, 0, r26
	dcmplti32	r19, 28, r19
	dcsel	r27, -1, r19
	dcmpeq32	r24, 0, r27
	daddi32	[rp4], r23, r23
	dandb	r19, -16, r27
	dshlb	r24, 4, r24
	dcsel	r27, r19, r27
	daddi32	r23, r24, r23
	dcmplt32	r9, r5, r24
	dcmplt32	25, r18, r18
	dcmplt32	r7, r9, r9
	dstcr	0x61, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcmpneq32	r22, 0, r19
	daddi32	[rp4 + 3], r25, r22
	dsubi32	4, r21, r21
	dmuli32	r22, r24, r22
	dshlb	r18, 4, r18
	dshlb	r9, 2, r9
	dcsel	r21, 4, r19
	dxorb	r9, 20, r9
	daddi32	r22, r18, r18
	dsubi32	4, r19, r21
	dmin32	r18, r9, r9
	dcp	r26, pls.maskh, south
	daddi32	r9, r21, r18
	dcp	r27, pls.maskl, south
	dcp	r23, pls.addr, south
	dcp	r18, pls.count1, south
	dsubi32	20, r9, r9
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r19, 0, :.LBB1_36
// %bb.2:                               //   in Loop: Header=BB1_1 Depth=1
	dstcr	1, r20
	djmpeqoff	r18, 0, :.LBB1_21
// %bb.3:                               //   in Loop: Header=BB1_1 Depth=1
	dstcr	1, r21
	dstcr	0, r20
	cp	crp2, crp3
	djmpeqoff	0, r9, :.LBB1_13
// %bb.4:                               //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
.LBB1_5:                                // %.preheader46
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_6 Depth 3
                                        //       Child Loop BB1_8 Depth 3
                                        //       Child Loop BB1_10 Depth 3
	dcp	r19, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB1_6
.LBB1_6:                                //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB1_5 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB1_8
.LBB1_8:                                //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB1_5 Depth=2
	dcp	r9, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB1_10
.LBB1_10:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB1_5 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r20, 3, :.LBB1_5
// %bb.12:                              // %Flow5
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r21
.LBB1_13:                               // %Flow7
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r20
	cp	crp2, crp3
	djmpeqoff	r21, 0, :.LBB1_20
// %bb.14:                              //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
.LBB1_15:                               // %.preheader44
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_16 Depth 3
                                        //       Child Loop BB1_18 Depth 3
	dcp	r19, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB1_16
.LBB1_16:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB1_15 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB1_18
.LBB1_18:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB1_15 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r20, 3, :.LBB1_15
.LBB1_20:                               // %Flow8
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r20
.LBB1_21:                               // %Flow13
                                        //   in Loop: Header=BB1_1 Depth=1
	djmpeqoff	r20, 0, :.LBB1_35
// %bb.22:                              //   in Loop: Header=BB1_1 Depth=1
	dstcr	1, r21
	dstcr	0, r20
	cp	crp2, crp3
	djmpeqoff	0, r9, :.LBB1_30
// %bb.23:                              //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB1_24:                               // %.preheader42
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_25 Depth 3
                                        //       Child Loop BB1_27 Depth 3
	dcp	r19, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB1_25
.LBB1_25:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB1_24 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB1_27
.LBB1_27:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB1_24 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r20, 3, :.LBB1_24
// %bb.29:                              // %Flow9
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r21
.LBB1_30:                               // %Flow11
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r20
	cp	crp2, crp3
	djmpeqoff	r21, 0, :.LBB1_35
// %bb.31:                              //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB1_32:                               // %.preheader40
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_33 Depth 3
	dcp	r19, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB1_33
.LBB1_33:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB1_32 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r20, 3, :.LBB1_32
.LBB1_35:                               // %Flow14
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r20
.LBB1_36:                               // %Flow25
                                        //   in Loop: Header=BB1_1 Depth=1
	djmpeqoff	r20, 0, :.LBB1_62
// %bb.37:                              //   in Loop: Header=BB1_1 Depth=1
	dstcr	1, r19
	djmpeqoff	r18, 0, :.LBB1_52
// %bb.38:                              //   in Loop: Header=BB1_1 Depth=1
	addi32	crp1, 16, crp3          //      
	dstcr	1, r20
	dstcr	0, r19
	cp	crp3, crp4
	djmpeqoff	0, r9, :.LBB1_46
// %bb.39:                              //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
.LBB1_40:                               // %.preheader38
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_41 Depth 3
                                        //       Child Loop BB1_43 Depth 3
	dcp	r18, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB1_41
.LBB1_41:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB1_40 Depth=2
	dcp	r9, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB1_43
.LBB1_43:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB1_40 Depth=2
	cp	south.0z, [crp4.z+=1]
	djmpincne	r19, 3, :.LBB1_40
// %bb.45:                              // %Flow15
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r20
.LBB1_46:                               // %Flow17
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r19
	djmpeqoff	r20, 0, :.LBB1_51
// %bb.47:                              //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB1_48:                               // %.preheader36
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_49 Depth 3
	dcp	r18, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB1_49
.LBB1_49:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB1_48 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r19, 3, :.LBB1_48
.LBB1_51:                               // %Flow18
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r19
.LBB1_52:                               // %Flow23
                                        //   in Loop: Header=BB1_1 Depth=1
	djmpeqoff	r19, 0, :.LBB1_62
// %bb.53:                              //   in Loop: Header=BB1_1 Depth=1
	addi32	crp1, 16, crp3          //      
	dstcr	1, r19
	dstcr	0, r18
	cp	crp3, crp4
	djmpeqoff	0, r9, :.LBB1_59
// %bb.54:                              //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB1_55:                               // %.preheader34
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_56 Depth 3
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB1_56
.LBB1_56:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB1_55 Depth=2
	cp	south.0z, [crp4.z+=1]
	djmpincne	r18, 3, :.LBB1_55
// %bb.58:                              // %Flow19
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0, r19
.LBB1_59:                               // %Flow21
                                        //   in Loop: Header=BB1_1 Depth=1
	djmpeqoff	r19, 0, :.LBB1_62
// %bb.60:                              //   in Loop: Header=BB1_1 Depth=1
	stcr	0x0, bitwidthmode
	djmpincsetup	0, 3, :.LBB1_61
.LBB1_61:                               // %.preheader
                                        //   Parent Loop BB1_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south.0z, [crp3.z+=1]
.LBB1_62:                               // %.loopexit
                                        //   in Loop: Header=BB1_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r9
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r9, pls.addr, west
	cp	crp1, crp3
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
	djmpincsetup	0, 16, :.LBB1_63
.LBB1_63:                               //   Parent Loop BB1_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp	crp2, crp4
	stcr	0x0, accumall
	stcr	0x2, bitwidthmode
	macwrxi8	[crp4+=2]
	stcr	0x1, bitwidthmode
	addi32	crp4, -8, crp4
	nrb	[crp4.z+=1], north | south | east | west
	macwrni8		<>	nnbr	r90
	macwrni8		<>	nrb	[crp4.z+=1], north | south | east | west
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{22}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	muli32lohi{19}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{15}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	nrb	cr6, north
	maxi32	south, cr6, cr6
	nrb	cr6, west
	maxi32	east, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{12}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb.lb	cr6, 16, [crp3.z+=1]
// %bb.64:                              //   in Loop: Header=BB1_1 Depth=1
	dcp	[rp2 + 1], r9
	cp	row, cr6
	dshlb	r2, 3, r2
	shrlb	cr6, 31, cr28
	cp	col, cr7
	dandb	r2, r28, r2
	addi32	cr6, cr28, cr28
	dshlb	r8, 3, r8
	shrlb	cr7, 31, cr29
	shrab	cr28, 1, cr28
	daddi32	r2, 8, r18
	dandb	r8, r28, r8
	orb	cr7, cr6, cr6
	addi32	cr7, cr29, cr7
	dcpc	r2, cr29
	addi32	cr28, cr29, cr28
	shrab	cr7, 1, cr7
	daddi32	r8, 8, r2
	dcpc	r18, cr29
	cmpltei32	cr29, cr28, cr29
	cmplti32	207, cr28, cr30
	dcpc	r8, cr31
	addi32	cr7, cr31, cr7
	orb	cr30, cr29, cr29
	dcpc	r2, cr30
	cmpltei32	cr30, cr7, cr30
	cmplti32	207, cr7, cr31
	muli32	cr28, 208, cr28
	orb	cr31, cr30, cr30
	andb	cr6, 1, cr6
	orb	cr29, cr30, cr29
	dcp	rp5, r2
	addi32	cr28, cr7, cr7
	xorb	cr29, 1, cr28
	dcp	[rp2], pls.addr, north
	dcpc	rp5, crp3
	dstcr	0x1, pc.resetfifo, north
	stcr	0x2, bitwidthmode
.LBB1_65:                               //   Parent Loop BB1_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB1_68 Depth 3
                                        //       Child Loop BB1_70 Depth 3
                                        //       Child Loop BB1_72 Depth 3
                                        //       Child Loop BB1_74 Depth 3
	cmpeq32	cr6, 0, cr29
	predpush	cr29, :.LBB1_77
// %bb.66:                              //   in Loop: Header=BB1_65 Depth=2
	predpush	cr28, :.LBB1_76
// %bb.67:                              //   in Loop: Header=BB1_65 Depth=2
	dmuli32	r2, r29, r8
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcpc	r8, cr29
	dcp	r9, dependencyid
	addi32	cr7, cr29, cr29
	dstcr	0x1, plsstatus, north
	dcp	flowid, r9
	shlb	cr29, 2, cr29
	djmpincsetup	0, 4, :.LBB1_68
	dstcr	0x260, pc.mode, north
	nrb	cr29, north
.LBB1_68:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.69:                              //   in Loop: Header=BB1_65 Depth=2
	djmpincsetup	0, 16, :.LBB1_70
	dstcr	0x360, pc.mode, north
.LBB1_70:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.71:                              //   in Loop: Header=BB1_65 Depth=2
	shlb	crp3, 2, crp4
	cp	crp1, crp5
	dstcr	0x260, pc.mode, north
	addi32	crp5, crp4, crp4
	djmpincsetup	0, 4, :.LBB1_72
	nrb	[crp4], north
.LBB1_72:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB1_65 Depth=2
	djmpincsetup	0, 16, :.LBB1_74
	dstcr	0x360, pc.mode, north
.LBB1_74:                               //   Parent Loop BB1_1 Depth=1
                                        //     Parent Loop BB1_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.75:                              //   in Loop: Header=BB1_65 Depth=2
	dstcr	0x260, pc.mode, north
.LBB1_76:                               // %Flow
                                        //   in Loop: Header=BB1_65 Depth=2
	predpop	
	addi32	crp3, 1, crp3
.LBB1_77:                               // %Flow4
                                        //   in Loop: Header=BB1_65 Depth=2
	predpop	
	djmpincne	r2, 4, :.LBB1_65
// %bb.78:                              //   in Loop: Header=BB1_1 Depth=1
	dstcr	-300, r31
	djmpincne	r10, r30, r31
.LBB1_79:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r27
	dcp	[rp1 + 2], r26
	dcp	[rp1 + 4], r25
	dcp	[rp1 + 6], r24
	dcp	[rp2], r23
	daddi32	rp1, 40, rp2
	dcp	[rp2], r22
	daddi32	rp1, 48, rp2
	dcp	[rp2], r21
	daddi32	rp1, 56, rp2
	dcp	[rp2], r20
	daddi32	rp1, 64, rp2
	dcp	[rp2], r19
	daddi32	rp1, 72, rp2
	dcp	[rp2], r18
	daddi32	rp1, 80, rp2
	dcp	[rp2], r9
	daddi32	rp1, 88, rp2
	dcp	[rp2], r8
	daddi32	rp1, 96, rp1
	addi32	crp1, 24, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj4ELj208ELj208EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj5120EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj8ELj104ELj104EEEEvRT0_RT1_RT2_
_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj4ELj208ELj208EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj5120EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj8ELj104ELj104EEEEvRT0_RT1_RT2_: // @_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj4ELj208ELj208EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj5120EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj8ELj104ELj104EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -24, rp1
	addi32	crp1, -48, crp1         //     
	dcp	r12, rp2
	dcp	r11, rp3
	dcp	r10, rp4
	dstcr	0, r10
	dstcr	1321528399, r11
	dstcr	832, r12
	dstcr	-1, r13
	addi32	crp1, 32, crp2          //      
	stcr	14566, cr10
	stcr	15154, cr11
	stcr	14664, cr12
	stcr	6553, cr13
	stcr	10559, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	0, rp5
	dstcr	11648, r14
	dstcr	0x2, mode
	dcp	r8, [rp1 + 4]
	dcp	r9, [rp1 + 2]
	dcp	r18, [rp1]
	dstcr	0xd0, pls.stride1, south
	dstcr	0x4, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xa900, pls.stride2, south
	dstcr	0x0, plsthresholdnorth
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x280, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB2_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_5 Depth 2
                                        //       Child Loop BB2_6 Depth 3
                                        //       Child Loop BB2_8 Depth 3
                                        //       Child Loop BB2_10 Depth 3
                                        //     Child Loop BB2_15 Depth 2
                                        //       Child Loop BB2_16 Depth 3
                                        //       Child Loop BB2_18 Depth 3
                                        //     Child Loop BB2_24 Depth 2
                                        //       Child Loop BB2_25 Depth 3
                                        //       Child Loop BB2_27 Depth 3
                                        //     Child Loop BB2_32 Depth 2
                                        //       Child Loop BB2_33 Depth 3
                                        //     Child Loop BB2_40 Depth 2
                                        //       Child Loop BB2_41 Depth 3
                                        //       Child Loop BB2_43 Depth 3
                                        //     Child Loop BB2_48 Depth 2
                                        //       Child Loop BB2_49 Depth 3
                                        //     Child Loop BB2_55 Depth 2
                                        //       Child Loop BB2_56 Depth 3
                                        //     Child Loop BB2_61 Depth 2
                                        //     Child Loop BB2_63 Depth 2
                                        //       Child Loop BB2_64 Depth 3
                                        //     Child Loop BB2_67 Depth 2
                                        //       Child Loop BB2_70 Depth 3
                                        //       Child Loop BB2_72 Depth 3
                                        //       Child Loop BB2_74 Depth 3
                                        //       Child Loop BB2_76 Depth 3
	dandb	r10, 255, r17
	daddi32	r10, 87, r15
	dmul32hi	r17, r11, r16
	dcmplt32	r17, 13, r5
	dandb	r15, 255, r5
	dshrlb	r16, 2, r15
	dstcr	1, r7
	dmuli32	r15, -13, r16
	dshlb	r15, 4, r6
	daddi32	r16, r10, r16
	dcsel	r6, 4, r28
	dshlb	r16, 4, r2
	dsubi32	r6, r28, r30
	dandb	r2, 240, r2
	dsubi32	192, r6, r6
	dsubi32	208, r2, r2
	dmaxi32	r6, 0, r8
	dmini32	r2, 20, r6
	dcmplt32	12, r17, r29
	dcsel	r30, 0, r30
	dsubi32	60, r6, r9
	dcmplti32	28, r6, r18
	dsubi32	28, r6, r18
	dshrlb	r13, r9, r9
	dshrlb	r13, r18, r18
	dmuli32	r30, r12, r30
	dandb	r16, 255, r2
	dcsel	r9, 0, r9
	dcmplti32	r6, 28, r6
	dcsel	r18, -1, r6
	dcmpeq32	r2, 0, r18
	daddi32	[rp4], r30, r30
	dandb	r6, -16, r18
	dshlb	r2, 6, r2
	dcsel	r18, r6, r18
	daddi32	r30, r2, r30
	dcmplt32	r17, 156, r2
	dcmplt32	12, r5, r5
	dcmplt32	155, r17, r17
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcmpneq32	r29, 0, r6
	daddi32	[rp4 + 3], r8, r29
	dsubi32	4, r28, r28
	dmuli32	r29, r2, r29
	dshlb	r5, 4, r5
	dshlb	r17, 2, r17
	dcsel	r28, 4, r6
	dxorb	r17, 20, r17
	daddi32	r29, r5, r5
	dsubi32	4, r6, r28
	dmin32	r5, r17, r17
	dcp	r9, pls.maskh, south
	daddi32	r17, r28, r5
	dcp	r18, pls.maskl, south
	dcp	r30, pls.addr, south
	dcp	r5, pls.count1, south
	dsubi32	20, r17, r17
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r6, 0, :.LBB2_36
// %bb.2:                               //   in Loop: Header=BB2_1 Depth=1
	dstcr	1, r7
	djmpeqoff	r5, 0, :.LBB2_21
// %bb.3:                               //   in Loop: Header=BB2_1 Depth=1
	dstcr	1, r28
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	0, r17, :.LBB2_13
// %bb.4:                               //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB2_5:                                // %.preheader46
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_6 Depth 3
                                        //       Child Loop BB2_8 Depth 3
                                        //       Child Loop BB2_10 Depth 3
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB2_6
.LBB2_6:                                //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB2_5 Depth=2
	dcp	r5, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB2_8
.LBB2_8:                                //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB2_5 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB2_10
.LBB2_10:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB2_5 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 4, :.LBB2_5
// %bb.12:                              // %Flow5
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r28
.LBB2_13:                               // %Flow7
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	r28, 0, :.LBB2_20
// %bb.14:                              //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB2_15:                               // %.preheader44
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_16 Depth 3
                                        //       Child Loop BB2_18 Depth 3
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB2_16
.LBB2_16:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB2_15 Depth=2
	dcp	r5, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB2_18
.LBB2_18:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB2_15 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 4, :.LBB2_15
.LBB2_20:                               // %Flow8
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r7
.LBB2_21:                               // %Flow13
                                        //   in Loop: Header=BB2_1 Depth=1
	djmpeqoff	r7, 0, :.LBB2_35
// %bb.22:                              //   in Loop: Header=BB2_1 Depth=1
	dstcr	1, r28
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	0, r17, :.LBB2_30
// %bb.23:                              //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB2_24:                               // %.preheader42
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_25 Depth 3
                                        //       Child Loop BB2_27 Depth 3
	dcp	r6, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB2_25
.LBB2_25:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB2_24 Depth=2
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB2_27
.LBB2_27:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB2_24 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 4, :.LBB2_24
// %bb.29:                              // %Flow9
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r28
.LBB2_30:                               // %Flow11
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	r28, 0, :.LBB2_35
// %bb.31:                              //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB2_32:                               // %.preheader40
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_33 Depth 3
	dcp	r6, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB2_33
.LBB2_33:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB2_32 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 4, :.LBB2_32
.LBB2_35:                               // %Flow14
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r7
.LBB2_36:                               // %Flow25
                                        //   in Loop: Header=BB2_1 Depth=1
	djmpeqoff	r7, 0, :.LBB2_62
// %bb.37:                              //   in Loop: Header=BB2_1 Depth=1
	dstcr	1, r6
	djmpeqoff	r5, 0, :.LBB2_52
// %bb.38:                              //   in Loop: Header=BB2_1 Depth=1
	addi32	crp1, 32, crp3          //      
	dstcr	1, r7
	dstcr	0, r6
	cp	crp3, crp4
	djmpeqoff	0, r17, :.LBB2_46
// %bb.39:                              //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB2_40:                               // %.preheader38
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_41 Depth 3
                                        //       Child Loop BB2_43 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB2_41
.LBB2_41:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB2_40 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB2_43
.LBB2_43:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB2_40 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r6, 4, :.LBB2_40
// %bb.45:                              // %Flow15
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r7
.LBB2_46:                               // %Flow17
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r6
	djmpeqoff	r7, 0, :.LBB2_51
// %bb.47:                              //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB2_48:                               // %.preheader36
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_49 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB2_49
.LBB2_49:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB2_48 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, 4, :.LBB2_48
.LBB2_51:                               // %Flow18
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r6
.LBB2_52:                               // %Flow23
                                        //   in Loop: Header=BB2_1 Depth=1
	djmpeqoff	r6, 0, :.LBB2_62
// %bb.53:                              //   in Loop: Header=BB2_1 Depth=1
	addi32	crp1, 32, crp3          //      
	dstcr	1, r6
	dstcr	0, r5
	cp	crp3, crp4
	djmpeqoff	0, r17, :.LBB2_59
// %bb.54:                              //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB2_55:                               // %.preheader34
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_56 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB2_56
.LBB2_56:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB2_55 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r5, 4, :.LBB2_55
// %bb.58:                              // %Flow19
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0, r6
.LBB2_59:                               // %Flow21
                                        //   in Loop: Header=BB2_1 Depth=1
	djmpeqoff	r6, 0, :.LBB2_62
// %bb.60:                              //   in Loop: Header=BB2_1 Depth=1
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 4, :.LBB2_61
.LBB2_61:                               // %.preheader
                                        //   Parent Loop BB2_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp3+=1]
.LBB2_62:                               // %.loopexit
                                        //   in Loop: Header=BB2_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r5
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r5, pls.addr, west
	dstcr	0, r17
	cp	crp1, crp3
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB2_63:                               //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_64 Depth 3
	cp	crp2, crp4
	stcr	0x0, accumall
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 7, :.LBB2_64
	macwrxi8	[crp4+=2]
	macwrxi8	[crp4+=2]
	stcr	0x1, bitwidthmode
	addi32	crp4, -16, crp4
	nrb	[crp4.z+=1], north | south | east | west
.LBB2_64:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp4.z+=1], north | south | east | west
// %bb.65:                              //   in Loop: Header=BB2_63 Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{22}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	muli32lohi{20}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{15}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	nrb	cr6, north
	maxi32	south, cr6, cr6
	nrb	cr6, west
	maxi32	east, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{10}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp3.z+=1]
	djmpincne	r17, 32, :.LBB2_63
// %bb.66:                              //   in Loop: Header=BB2_1 Depth=1
	dcp	[rp2 + 1], r17
	cp	row, cr6
	dshlb	r15, 3, r15
	shrlb	cr6, 31, cr28
	cp	col, cr7
	addi32	cr6, cr28, cr28
	dandb	r15, 248, r15
	shrlb	cr7, 31, cr29
	dshlb	r16, 3, r16
	shrab	cr28, 1, cr28
	daddi32	r15, 8, r5
	addi32	cr7, cr29, cr29
	dandb	r16, 248, r16
	orb	cr7, cr6, cr6
	dcpc	r15, cr7
	addi32	cr28, cr7, cr7
	shrab	cr29, 1, cr29
	daddi32	r16, 8, r15
	dcpc	r5, cr28
	cmpltei32	cr28, cr7, cr28
	cmplti32	103, cr7, cr30
	dcpc	r16, cr31
	addi32	cr29, cr31, cr29
	orb	cr30, cr28, cr28
	dcpc	r15, cr30
	cmpltei32	cr30, cr29, cr30
	cmplti32	103, cr29, cr31
	muli32	cr7, 112, cr7
	orb	cr31, cr30, cr30
	andb	cr6, 1, cr6
	orb	cr28, cr30, cr28
	addi32	cr7, cr29, cr7
	xorb	cr28, 1, cr28
	dcp	rp5, r15
	dcp	[rp2], pls.addr, north
	dcpc	rp5, crp3
	stcr	0x2, bitwidthmode
	dstcr	0x1, pc.resetfifo, north
.LBB2_67:                               //   Parent Loop BB2_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB2_70 Depth 3
                                        //       Child Loop BB2_72 Depth 3
                                        //       Child Loop BB2_74 Depth 3
                                        //       Child Loop BB2_76 Depth 3
	cmpeq32	cr6, 0, cr29
	predpush	cr29, :.LBB2_79
// %bb.68:                              //   in Loop: Header=BB2_67 Depth=2
	predpush	cr28, :.LBB2_78
// %bb.69:                              //   in Loop: Header=BB2_67 Depth=2
	dmuli32	r15, r14, r16
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcpc	r16, cr29
	dcp	r17, dependencyid
	addi32	cr7, cr29, cr29
	dstcr	0x1, plsstatus, north
	dcp	flowid, r17
	shlb	cr29, 2, cr29
	djmpincsetup	0, 4, :.LBB2_70
	dstcr	0x260, pc.mode, north
	nrb	cr29, north
.LBB2_70:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_67 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.71:                              //   in Loop: Header=BB2_67 Depth=2
	djmpincsetup	0, 16, :.LBB2_72
	dstcr	0x360, pc.mode, north
.LBB2_72:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_67 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB2_67 Depth=2
	shlb	crp3, 2, crp4
	cp	crp1, crp5
	dstcr	0x260, pc.mode, north
	addi32	crp5, crp4, crp4
	djmpincsetup	0, 4, :.LBB2_74
	nrb	[crp4], north
.LBB2_74:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_67 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.75:                              //   in Loop: Header=BB2_67 Depth=2
	djmpincsetup	0, 16, :.LBB2_76
	dstcr	0x360, pc.mode, north
.LBB2_76:                               //   Parent Loop BB2_1 Depth=1
                                        //     Parent Loop BB2_67 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.77:                              //   in Loop: Header=BB2_67 Depth=2
	dstcr	0x260, pc.mode, north
.LBB2_78:                               // %Flow
                                        //   in Loop: Header=BB2_67 Depth=2
	predpop	
	addi32	crp3, 1, crp3
.LBB2_79:                               // %Flow4
                                        //   in Loop: Header=BB2_67 Depth=2
	predpop	
	djmpincne	r15, 8, :.LBB2_67
// %bb.80:                              //   in Loop: Header=BB2_1 Depth=1
	dstcr	-303, r31
	djmpincne	r10, 169, r31
.LBB2_81:
	dcp	[rp1], r18
	dcp	[rp1 + 2], r9
	dcp	[rp1 + 4], r8
	daddi32	rp1, 24, rp1
	addi32	crp1, 48, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj8ELj104ELj104EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj19456EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj16ELj52ELj52EEEEvRT0_RT1_RT2_
_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj8ELj104ELj104EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj19456EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj16ELj52ELj52EEEEvRT0_RT1_RT2_: // @_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj8ELj104ELj104EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj19456EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj16ELj52ELj52EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -24, rp1
	addi32	crp1, -96, crp1         //     
	dcp	r12, rp2
	dcp	r11, rp3
	dcp	r10, rp4
	dstcr	0, r10
	dstcr	613566757, r11
	dstcr	448, r12
	dstcr	-1, r13
	addi32	crp1, 64, crp2          //      
	stcr	13691, cr10
	stcr	15662, cr11
	stcr	8480, cr12
	stcr	6553, cr13
	stcr	10762, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	0, rp5
	dstcr	3328, r14
	dstcr	0x2, mode
	dcp	r8, [rp1 + 4]
	dcp	r9, [rp1 + 2]
	dcp	r18, [rp1]
	dstcr	0x70, pls.stride1, south
	dstcr	0x8, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x2d80, pls.stride2, south
	dstcr	0x0, plsthresholdnorth
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x980, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB3_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB3_5 Depth 2
                                        //       Child Loop BB3_6 Depth 3
                                        //       Child Loop BB3_8 Depth 3
                                        //       Child Loop BB3_10 Depth 3
                                        //     Child Loop BB3_15 Depth 2
                                        //       Child Loop BB3_16 Depth 3
                                        //       Child Loop BB3_18 Depth 3
                                        //     Child Loop BB3_24 Depth 2
                                        //       Child Loop BB3_25 Depth 3
                                        //       Child Loop BB3_27 Depth 3
                                        //     Child Loop BB3_32 Depth 2
                                        //       Child Loop BB3_33 Depth 3
                                        //     Child Loop BB3_40 Depth 2
                                        //       Child Loop BB3_41 Depth 3
                                        //       Child Loop BB3_43 Depth 3
                                        //     Child Loop BB3_48 Depth 2
                                        //       Child Loop BB3_49 Depth 3
                                        //     Child Loop BB3_55 Depth 2
                                        //       Child Loop BB3_56 Depth 3
                                        //     Child Loop BB3_61 Depth 2
                                        //     Child Loop BB3_63 Depth 2
                                        //       Child Loop BB3_64 Depth 3
                                        //       Child Loop BB3_66 Depth 3
                                        //     Child Loop BB3_69 Depth 2
                                        //       Child Loop BB3_72 Depth 3
                                        //       Child Loop BB3_74 Depth 3
                                        //       Child Loop BB3_76 Depth 3
                                        //       Child Loop BB3_78 Depth 3
	dandb	r10, 255, r17
	dstcr	1, r7
	dmul32hi	r17, r11, r15
	dcmplt32	r17, 7, r16
	dsubi32	r17, r15, r16
	dshrlb	r16, 1, r16
	daddi32	r16, r15, r15
	dshrlb	r15, 2, r15
	dmuli32	r15, -7, r16
	dshlb	r15, 4, r5
	daddi32	r16, r10, r16
	dcsel	r5, 4, r6
	dshlb	r16, 4, r30
	dsubi32	r5, r6, r29
	dandb	r30, 240, r30
	dcmplt32	6, r17, r28
	dsubi32	104, r30, r30
	dcsel	r29, 0, r29
	dmini32	r30, 20, r30
	dmuli32	r29, r12, r29
	dsubi32	60, r30, r9
	dcmplti32	28, r30, r18
	dshrlb	r13, r9, r9
	dsubi32	28, r30, r18
	dcsel	r9, 0, r9
	dcmplti32	r30, 28, r30
	dshrlb	r13, r18, r30
	dandb	r16, 255, r8
	dsubi32	88, r5, r2
	dcsel	r30, -1, r30
	dshlb	r8, 6, r18
	dcmpeq32	r8, 0, r8
	daddi32	[rp4], r29, r29
	dmaxi32	r2, 0, r2
	dandb	r30, -16, r8
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	daddi32	r29, r18, r29
	dcsel	r8, r30, r30
	dcmplt32	r17, 42, r8
	daddi32	[rp4 + 3], r2, r2
	dsubi32	104, r5, r5
	dcmplt32	41, r17, r17
	dcp	r9, pls.maskh, south
	dcp	r30, pls.maskl, south
	dcp	r29, pls.addr, south
	dmuli32	r2, r8, r29
	dmin32	r5, 16, r5
	dsubi32	4, r6, r6
	dshlb	r17, 2, r17
	dcmpneq32	r28, 0, r28
	dcsel	r6, 4, r6
	dxorb	r17, 20, r17
	daddi32	r29, r5, r5
	dsubi32	4, r6, r28
	dmin32	r5, r17, r17
	daddi32	r17, r28, r5
	dsubi32	20, r17, r17
	dcp	r5, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r6, 0, :.LBB3_36
// %bb.2:                               //   in Loop: Header=BB3_1 Depth=1
	dstcr	1, r7
	djmpeqoff	r5, 0, :.LBB3_21
// %bb.3:                               //   in Loop: Header=BB3_1 Depth=1
	dstcr	1, r28
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	0, r17, :.LBB3_13
// %bb.4:                               //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB3_5:                                // %.preheader46
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_6 Depth 3
                                        //       Child Loop BB3_8 Depth 3
                                        //       Child Loop BB3_10 Depth 3
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB3_6
.LBB3_6:                                //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB3_5 Depth=2
	dcp	r5, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB3_8
.LBB3_8:                                //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB3_5 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB3_10
.LBB3_10:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB3_5 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 8, :.LBB3_5
// %bb.12:                              // %Flow5
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r28
.LBB3_13:                               // %Flow7
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	r28, 0, :.LBB3_20
// %bb.14:                              //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB3_15:                               // %.preheader44
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_16 Depth 3
                                        //       Child Loop BB3_18 Depth 3
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB3_16
.LBB3_16:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB3_15 Depth=2
	dcp	r5, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB3_18
.LBB3_18:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB3_15 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 8, :.LBB3_15
.LBB3_20:                               // %Flow8
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r7
.LBB3_21:                               // %Flow13
                                        //   in Loop: Header=BB3_1 Depth=1
	djmpeqoff	r7, 0, :.LBB3_35
// %bb.22:                              //   in Loop: Header=BB3_1 Depth=1
	dstcr	1, r28
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	0, r17, :.LBB3_30
// %bb.23:                              //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB3_24:                               // %.preheader42
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_25 Depth 3
                                        //       Child Loop BB3_27 Depth 3
	dcp	r6, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB3_25
.LBB3_25:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB3_24 Depth=2
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB3_27
.LBB3_27:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB3_24 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 8, :.LBB3_24
// %bb.29:                              // %Flow9
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r28
.LBB3_30:                               // %Flow11
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r7
	cp	crp2, crp3
	djmpeqoff	r28, 0, :.LBB3_35
// %bb.31:                              //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB3_32:                               // %.preheader40
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_33 Depth 3
	dcp	r6, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB3_33
.LBB3_33:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB3_32 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r7, 8, :.LBB3_32
.LBB3_35:                               // %Flow14
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r7
.LBB3_36:                               // %Flow25
                                        //   in Loop: Header=BB3_1 Depth=1
	djmpeqoff	r7, 0, :.LBB3_62
// %bb.37:                              //   in Loop: Header=BB3_1 Depth=1
	dstcr	1, r6
	djmpeqoff	r5, 0, :.LBB3_52
// %bb.38:                              //   in Loop: Header=BB3_1 Depth=1
	addi32	crp1, 64, crp3          //      
	dstcr	1, r7
	dstcr	0, r6
	cp	crp3, crp4
	djmpeqoff	0, r17, :.LBB3_46
// %bb.39:                              //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB3_40:                               // %.preheader38
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_41 Depth 3
                                        //       Child Loop BB3_43 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB3_41
.LBB3_41:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB3_40 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB3_43
.LBB3_43:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB3_40 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r6, 8, :.LBB3_40
// %bb.45:                              // %Flow15
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r7
.LBB3_46:                               // %Flow17
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r6
	djmpeqoff	r7, 0, :.LBB3_51
// %bb.47:                              //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB3_48:                               // %.preheader36
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_49 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB3_49
.LBB3_49:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB3_48 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, 8, :.LBB3_48
.LBB3_51:                               // %Flow18
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r6
.LBB3_52:                               // %Flow23
                                        //   in Loop: Header=BB3_1 Depth=1
	djmpeqoff	r6, 0, :.LBB3_62
// %bb.53:                              //   in Loop: Header=BB3_1 Depth=1
	addi32	crp1, 64, crp3          //      
	dstcr	1, r6
	dstcr	0, r5
	cp	crp3, crp4
	djmpeqoff	0, r17, :.LBB3_59
// %bb.54:                              //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB3_55:                               // %.preheader34
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_56 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB3_56
.LBB3_56:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB3_55 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r5, 8, :.LBB3_55
// %bb.58:                              // %Flow19
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0, r6
.LBB3_59:                               // %Flow21
                                        //   in Loop: Header=BB3_1 Depth=1
	djmpeqoff	r6, 0, :.LBB3_62
// %bb.60:                              //   in Loop: Header=BB3_1 Depth=1
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 8, :.LBB3_61
.LBB3_61:                               // %.preheader
                                        //   Parent Loop BB3_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp3+=1]
.LBB3_62:                               // %.loopexit
                                        //   in Loop: Header=BB3_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r5
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r5, pls.addr, west
	dstcr	0, r17
	cp	crp1, crp3
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB3_63:                               //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_64 Depth 3
                                        //       Child Loop BB3_66 Depth 3
	cp	crp2, crp4
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 4, :.LBB3_64
	stcr	0x0, accumall
.LBB3_64:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrxi8.lb	[crp4+=2]
// %bb.65:                              //   in Loop: Header=BB3_63 Depth=2
	addi32	crp4, -32, crp4
	stcr	0x1, bitwidthmode
	djmpincsetup	0, 15, :.LBB3_66
	nrb	[crp4.z+=1], north | south | east | west
.LBB3_66:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp4.z+=1], north | south | east | west
// %bb.67:                              //   in Loop: Header=BB3_63 Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{22}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	muli32lohi{20}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{15}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	nrb	cr6, north
	maxi32	south, cr6, cr6
	nrb	cr6, west
	maxi32	east, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{10}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp3.z+=1]
	djmpincne	r17, 64, :.LBB3_63
// %bb.68:                              //   in Loop: Header=BB3_1 Depth=1
	dcp	[rp2 + 1], r17
	cp	row, cr6
	cp	col, cr7
	shrlb	cr6, 31, cr28
	dshlb	r15, 3, r15
	addi32	cr6, cr28, cr28
	shrlb	cr7, 31, cr29
	dshlb	r16, 3, r16
	shrab	cr28, 1, cr28
	daddi32	r15, 8, r5
	addi32	cr7, cr29, cr29
	dandb	r16, 248, r16
	orb	cr7, cr6, cr7
	dcpc	r15, cr6
	addi32	cr28, cr6, cr6
	shrab	cr29, 1, cr29
	daddi32	r16, 8, r15
	dcpc	r5, cr28
	cmpltei32	cr28, cr6, cr28
	cmplti32	51, cr6, cr30
	dcpc	r16, cr31
	addi32	cr29, cr31, cr29
	orb	cr30, cr28, cr28
	dcpc	r15, cr30
	cmpltei32	cr30, cr29, cr30
	cmplti32	51, cr29, cr31
	shlb	cr6, 6, cr6
	orb	cr31, cr30, cr30
	addi32	cr6, cr29, cr6
	orb	cr28, cr30, cr28
	andb	cr7, 1, cr7
	xorb	cr28, 1, cr28
	dcp	rp5, r15
	dcp	[rp2], pls.addr, north
	dcpc	rp5, crp3
	stcr	0x2, bitwidthmode
	dstcr	0x1, pc.resetfifo, north
.LBB3_69:                               //   Parent Loop BB3_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB3_72 Depth 3
                                        //       Child Loop BB3_74 Depth 3
                                        //       Child Loop BB3_76 Depth 3
                                        //       Child Loop BB3_78 Depth 3
	cmpeq32	cr7, 0, cr29
	predpush	cr29, :.LBB3_81
// %bb.70:                              //   in Loop: Header=BB3_69 Depth=2
	predpush	cr28, :.LBB3_80
// %bb.71:                              //   in Loop: Header=BB3_69 Depth=2
	dmuli32	r15, r14, r16
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcpc	r16, cr29
	dcp	r17, dependencyid
	addi32	cr6, cr29, cr29
	dstcr	0x1, plsstatus, north
	dcp	flowid, r17
	shlb	cr29, 2, cr29
	djmpincsetup	0, 4, :.LBB3_72
	dstcr	0x260, pc.mode, north
	nrb	cr29, north
.LBB3_72:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB3_69 Depth=2
	djmpincsetup	0, 16, :.LBB3_74
	dstcr	0x360, pc.mode, north
.LBB3_74:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.75:                              //   in Loop: Header=BB3_69 Depth=2
	shlb	crp3, 2, crp4
	cp	crp1, crp5
	dstcr	0x260, pc.mode, north
	addi32	crp5, crp4, crp4
	djmpincsetup	0, 4, :.LBB3_76
	nrb	[crp4], north
.LBB3_76:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.77:                              //   in Loop: Header=BB3_69 Depth=2
	djmpincsetup	0, 16, :.LBB3_78
	dstcr	0x360, pc.mode, north
.LBB3_78:                               //   Parent Loop BB3_1 Depth=1
                                        //     Parent Loop BB3_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.79:                              //   in Loop: Header=BB3_69 Depth=2
	dstcr	0x260, pc.mode, north
.LBB3_80:                               // %Flow
                                        //   in Loop: Header=BB3_69 Depth=2
	predpop	
	addi32	crp3, 1, crp3
.LBB3_81:                               // %Flow4
                                        //   in Loop: Header=BB3_69 Depth=2
	predpop	
	djmpincne	r15, 16, :.LBB3_69
// %bb.82:                              //   in Loop: Header=BB3_1 Depth=1
	dstcr	-303, r31
	djmpincne	r10, 49, r31
.LBB3_83:
	dcp	[rp1], r18
	dcp	[rp1 + 2], r9
	dcp	[rp1 + 4], r8
	daddi32	rp1, 24, rp1
	addi32	crp1, 96, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj16ELj52ELj52EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj75776EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj32ELj26ELj26EEEEvRT0_RT1_RT2_
_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj16ELj52ELj52EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj75776EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj32ELj26ELj26EEEEvRT0_RT1_RT2_: // @_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj16ELj52ELj52EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj75776EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj32ELj26ELj26EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	addi32	crp1, -208, crp1        //     
	dcp	r12, rp2
	dcp	r11, rp3
	dcp	r10, rp4
	dstcr	0, r10
	dstcr	-1, r11
	addi32	crp1, 144, crp2         //      
	stcr	9716, cr10
	stcr	9130, cr11
	stcr	13400, cr12
	stcr	6553, cr13
	stcr	11992, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	0, rp5
	dstcr	832, r12
	dstcr	0x40, pls.stride1, south
	dstcr	0x10, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xd00, pls.stride2, south
	dstcr	0x0, plsthresholdnorth
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x2500, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x2, mode
	dstcr	0x10, pls.stride2, west
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB4_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB4_5 Depth 2
                                        //       Child Loop BB4_6 Depth 3
                                        //       Child Loop BB4_8 Depth 3
                                        //       Child Loop BB4_10 Depth 3
                                        //     Child Loop BB4_15 Depth 2
                                        //       Child Loop BB4_16 Depth 3
                                        //       Child Loop BB4_18 Depth 3
                                        //     Child Loop BB4_24 Depth 2
                                        //       Child Loop BB4_25 Depth 3
                                        //       Child Loop BB4_27 Depth 3
                                        //     Child Loop BB4_32 Depth 2
                                        //       Child Loop BB4_33 Depth 3
                                        //     Child Loop BB4_40 Depth 2
                                        //       Child Loop BB4_41 Depth 3
                                        //       Child Loop BB4_43 Depth 3
                                        //     Child Loop BB4_48 Depth 2
                                        //       Child Loop BB4_49 Depth 3
                                        //     Child Loop BB4_55 Depth 2
                                        //       Child Loop BB4_56 Depth 3
                                        //     Child Loop BB4_61 Depth 2
                                        //     Child Loop BB4_63 Depth 2
                                        //       Child Loop BB4_64 Depth 3
                                        //       Child Loop BB4_66 Depth 3
                                        //     Child Loop BB4_69 Depth 2
                                        //       Child Loop BB4_72 Depth 3
                                        //       Child Loop BB4_74 Depth 3
                                        //       Child Loop BB4_76 Depth 3
                                        //       Child Loop BB4_78 Depth 3
	dshrlb	r10, 2, r14
	dandb	r10, 3, r13
	dcmpeq32	r14, 0, r15
	dshlb	r14, 4, r15
	dshlb	r13, 4, r16
	dcsel	r15, 4, r17
	dcmpneq32	r14, 0, r5
	dsubi32	r15, r17, r6
	dsubi32	52, r16, r16
	dcsel	r6, 0, r6
	dmini32	r16, 20, r16
	dshlb	r6, 8, r6
	dshlb	r13, 6, r28
	daddi32	[rp4], r6, r6
	dsubi32	60, r16, r29
	daddi32	r6, r28, r6
	dshrlb	r11, r29, r28
	dcmplti32	28, r16, r29
	dsubi32	28, r16, r29
	dcsel	r28, 0, r28
	dcmplti32	r16, 28, r16
	dshrlb	r11, r29, r16
	dsubi32	36, r15, r7
	dcsel	r16, -1, r16
	dmaxi32	r7, 0, r7
	dandb	r16, -16, r30
	dcmpeq32	r13, 0, r2
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcsel	r30, r16, r16
	dcmplt32	r15, 36, r30
	daddi32	[rp4 + 3], r7, r7
	dsubi32	52, r15, r29
	dcmplt32	35, r15, r15
	dcp	r28, pls.maskh, south
	dcp	r16, pls.maskl, south
	dmuli32	r7, r30, r16
	dmin32	r29, 16, r29
	dsubi32	4, r17, r17
	dshlb	r15, 2, r15
	dcmpneq32	r5, 0, r5
	dcsel	r17, 4, r17
	dxorb	r15, 20, r15
	daddi32	r16, r29, r16
	dsubi32	4, r17, r2
	dmin32	r16, r15, r15
	dcp	r6, pls.addr, south
	daddi32	r15, r2, r16
	dstcr	1, r5
	dcp	r16, pls.count1, south
	dsubi32	20, r15, r15
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r17, 0, :.LBB4_36
// %bb.2:                               //   in Loop: Header=BB4_1 Depth=1
	dstcr	1, r5
	djmpeqoff	r16, 0, :.LBB4_21
// %bb.3:                               //   in Loop: Header=BB4_1 Depth=1
	dstcr	1, r6
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	0, r15, :.LBB4_13
// %bb.4:                               //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB4_5:                                // %.preheader46
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_6 Depth 3
                                        //       Child Loop BB4_8 Depth 3
                                        //       Child Loop BB4_10 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB4_6
.LBB4_6:                                //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB4_5 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB4_8
.LBB4_8:                                //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB4_5 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB4_10
.LBB4_10:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB4_5 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 16, :.LBB4_5
// %bb.12:                              // %Flow5
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r6
.LBB4_13:                               // %Flow7
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	r6, 0, :.LBB4_20
// %bb.14:                              //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB4_15:                               // %.preheader44
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_16 Depth 3
                                        //       Child Loop BB4_18 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB4_16
.LBB4_16:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB4_15 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB4_18
.LBB4_18:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB4_15 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 16, :.LBB4_15
.LBB4_20:                               // %Flow8
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r5
.LBB4_21:                               // %Flow13
                                        //   in Loop: Header=BB4_1 Depth=1
	djmpeqoff	r5, 0, :.LBB4_35
// %bb.22:                              //   in Loop: Header=BB4_1 Depth=1
	dstcr	1, r6
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	0, r15, :.LBB4_30
// %bb.23:                              //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB4_24:                               // %.preheader42
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_25 Depth 3
                                        //       Child Loop BB4_27 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB4_25
.LBB4_25:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB4_24 Depth=2
	dcp	r15, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB4_27
.LBB4_27:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB4_24 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 16, :.LBB4_24
// %bb.29:                              // %Flow9
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r6
.LBB4_30:                               // %Flow11
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	r6, 0, :.LBB4_35
// %bb.31:                              //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB4_32:                               // %.preheader40
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_33 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB4_33
.LBB4_33:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB4_32 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 16, :.LBB4_32
.LBB4_35:                               // %Flow14
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r5
.LBB4_36:                               // %Flow25
                                        //   in Loop: Header=BB4_1 Depth=1
	djmpeqoff	r5, 0, :.LBB4_62
// %bb.37:                              //   in Loop: Header=BB4_1 Depth=1
	dstcr	1, r17
	djmpeqoff	r16, 0, :.LBB4_52
// %bb.38:                              //   in Loop: Header=BB4_1 Depth=1
	addi32	crp1, 144, crp3         //      
	dstcr	1, r5
	dstcr	0, r17
	cp	crp3, crp4
	djmpeqoff	0, r15, :.LBB4_46
// %bb.39:                              //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB4_40:                               // %.preheader38
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_41 Depth 3
                                        //       Child Loop BB4_43 Depth 3
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB4_41
.LBB4_41:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB4_40 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB4_43
.LBB4_43:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB4_40 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r17, 16, :.LBB4_40
// %bb.45:                              // %Flow15
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r5
.LBB4_46:                               // %Flow17
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r17
	djmpeqoff	r5, 0, :.LBB4_51
// %bb.47:                              //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB4_48:                               // %.preheader36
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_49 Depth 3
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB4_49
.LBB4_49:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB4_48 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r17, 16, :.LBB4_48
.LBB4_51:                               // %Flow18
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r17
.LBB4_52:                               // %Flow23
                                        //   in Loop: Header=BB4_1 Depth=1
	djmpeqoff	r17, 0, :.LBB4_62
// %bb.53:                              //   in Loop: Header=BB4_1 Depth=1
	addi32	crp1, 144, crp3         //      
	dstcr	1, r17
	dstcr	0, r16
	cp	crp3, crp4
	djmpeqoff	0, r15, :.LBB4_59
// %bb.54:                              //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB4_55:                               // %.preheader34
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_56 Depth 3
	dcp	r15, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB4_56
.LBB4_56:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB4_55 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r16, 16, :.LBB4_55
// %bb.58:                              // %Flow19
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0, r17
.LBB4_59:                               // %Flow21
                                        //   in Loop: Header=BB4_1 Depth=1
	djmpeqoff	r17, 0, :.LBB4_62
// %bb.60:                              //   in Loop: Header=BB4_1 Depth=1
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 16, :.LBB4_61
.LBB4_61:                               // %.preheader
                                        //   Parent Loop BB4_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp3+=1]
.LBB4_62:                               // %.loopexit
                                        //   in Loop: Header=BB4_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r16
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r16, pls.addr, west
	dstcr	0, r15
	addi32	crp1, 16, crp3          //      
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB4_63:                               //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_64 Depth 3
                                        //       Child Loop BB4_66 Depth 3
	cp	crp2, crp4
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 8, :.LBB4_64
	stcr	0x0, accumall
.LBB4_64:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrxi8.lb	[crp4+=2]
// %bb.65:                              //   in Loop: Header=BB4_63 Depth=2
	addi32	crp4, -64, crp4
	stcr	0x1, bitwidthmode
	djmpincsetup	0, 31, :.LBB4_66
	nrb	[crp4.z+=1], north | south | east | west
.LBB4_66:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp4.z+=1], north | south | east | west
// %bb.67:                              //   in Loop: Header=BB4_63 Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{22}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	muli32lohi{19}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{16}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	nrb	cr6, north
	maxi32	south, cr6, cr6
	nrb	cr6, west
	maxi32	east, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{10}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp3.z+=1]
	djmpincne	r15, 128, :.LBB4_63
// %bb.68:                              //   in Loop: Header=BB4_1 Depth=1
	dcp	[rp2 + 1], r15
	cp	row, cr6
	cp	col, cr7
	shrlb	cr6, 31, cr28
	dshlb	r14, 3, r14
	addi32	cr6, cr28, cr28
	shrlb	cr7, 31, cr29
	shrab	cr28, 1, cr28
	daddi32	r14, 8, r16
	addi32	cr7, cr29, cr29
	dshlb	r13, 3, r13
	orb	cr7, cr6, cr7
	dcpc	r14, cr6
	addi32	cr28, cr6, cr6
	shrab	cr29, 1, cr29
	daddi32	r13, 8, r14
	dcpc	r16, cr28
	cmpltei32	cr28, cr6, cr28
	cmplti32	25, cr6, cr30
	dcpc	r13, cr31
	addi32	cr29, cr31, cr29
	orb	cr30, cr28, cr28
	dcpc	r14, cr30
	cmpltei32	cr30, cr29, cr30
	cmplti32	25, cr29, cr31
	shlb	cr6, 5, cr6
	orb	cr31, cr30, cr30
	addi32	cr6, cr29, cr6
	orb	cr28, cr30, cr28
	andb	cr7, 1, cr7
	xorb	cr28, 1, cr28
	dcp	rp5, r13
	dcp	[rp2], pls.addr, north
	dcpc	rp5, crp3
	stcr	0x2, bitwidthmode
	dstcr	0x1, pc.resetfifo, north
.LBB4_69:                               //   Parent Loop BB4_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB4_72 Depth 3
                                        //       Child Loop BB4_74 Depth 3
                                        //       Child Loop BB4_76 Depth 3
                                        //       Child Loop BB4_78 Depth 3
	cmpeq32	cr7, 0, cr29
	predpush	cr29, :.LBB4_81
// %bb.70:                              //   in Loop: Header=BB4_69 Depth=2
	predpush	cr28, :.LBB4_80
// %bb.71:                              //   in Loop: Header=BB4_69 Depth=2
	dmuli32	r13, r12, r14
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcpc	r14, cr29
	dcp	r15, dependencyid
	addi32	cr6, cr29, cr29
	dstcr	0x1, plsstatus, north
	dcp	flowid, r15
	shlb	cr29, 2, cr29
	djmpincsetup	0, 4, :.LBB4_72
	dstcr	0x260, pc.mode, north
	nrb	cr29, north
.LBB4_72:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB4_69 Depth=2
	djmpincsetup	0, 16, :.LBB4_74
	dstcr	0x360, pc.mode, north
.LBB4_74:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.75:                              //   in Loop: Header=BB4_69 Depth=2
	shlb	crp3, 2, crp4
	addi32	crp1, 16, crp5          //      
	dstcr	0x260, pc.mode, north
	addi32	crp5, crp4, crp4
	djmpincsetup	0, 4, :.LBB4_76
	nrb	[crp4], north
.LBB4_76:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.77:                              //   in Loop: Header=BB4_69 Depth=2
	djmpincsetup	0, 16, :.LBB4_78
	dstcr	0x360, pc.mode, north
.LBB4_78:                               //   Parent Loop BB4_1 Depth=1
                                        //     Parent Loop BB4_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.79:                              //   in Loop: Header=BB4_69 Depth=2
	dstcr	0x260, pc.mode, north
.LBB4_80:                               // %Flow
                                        //   in Loop: Header=BB4_69 Depth=2
	predpop	
	addi32	crp3, 1, crp3
.LBB4_81:                               // %Flow4
                                        //   in Loop: Header=BB4_69 Depth=2
	predpop	
	djmpincne	r13, 32, :.LBB4_69
// %bb.82:                              //   in Loop: Header=BB4_1 Depth=1
	dstcr	-294, r31
	djmpincne	r10, 16, r31
.LBB4_83:
	daddi32	rp1, 16, rp1
	addi32	crp1, 208, crp1         //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_8552357946429664381_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj32ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj299008EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj256ELj26ELj26EEEEvRT0_RT1_RT2_
_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_8552357946429664381_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj32ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj299008EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj256ELj26ELj26EEEEvRT0_RT1_RT2_: // @_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_8552357946429664381_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj32ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj299008EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj256ELj26ELj26EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -24, rp1
	stcr	-1168, cr10
	stcr	1040, crp2
	addi32	crp1, cr10, crp1        //     
	dcp	r12, rp2
	dcp	r11, rp3
	dcp	r10, rp4
	dstcr	0, r10
	dstcr	-1, r11
	addi32	crp1, crp2, crp2
	stcr	8315, cr10
	stcr	14621, cr11
	stcr	9226, cr12
	stcr	6553, cr13
	dstcr	256, r12
	dstcr	0x2, mode
	dcp	r8, [rp1 + 4]
	dstcr	0x20, pls.stride1, south
	dstcr	0x20, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x340, pls.stride2, south
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x9200, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x20, pls.stride1, north
	dstcr	0x100, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x340, pls.stride2, north
.LBB5_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_5 Depth 2
                                        //       Child Loop BB5_6 Depth 3
                                        //       Child Loop BB5_8 Depth 3
                                        //       Child Loop BB5_10 Depth 3
                                        //     Child Loop BB5_15 Depth 2
                                        //       Child Loop BB5_16 Depth 3
                                        //       Child Loop BB5_18 Depth 3
                                        //     Child Loop BB5_24 Depth 2
                                        //       Child Loop BB5_25 Depth 3
                                        //       Child Loop BB5_27 Depth 3
                                        //     Child Loop BB5_32 Depth 2
                                        //       Child Loop BB5_33 Depth 3
                                        //     Child Loop BB5_40 Depth 2
                                        //       Child Loop BB5_41 Depth 3
                                        //       Child Loop BB5_43 Depth 3
                                        //     Child Loop BB5_48 Depth 2
                                        //       Child Loop BB5_49 Depth 3
                                        //     Child Loop BB5_55 Depth 2
                                        //       Child Loop BB5_56 Depth 3
                                        //     Child Loop BB5_61 Depth 2
                                        //     Child Loop BB5_63 Depth 2
                                        //       Child Loop BB5_64 Depth 3
                                        //       Child Loop BB5_66 Depth 3
                                        //     Child Loop BB5_69 Depth 2
                                        //       Child Loop BB5_70 Depth 3
                                        //       Child Loop BB5_72 Depth 3
	dshrlb	r10, 1, r14
	dshlb	r10, 4, r15
	dcmpeq32	r14, 0, r13
	dshlb	r14, 4, r13
	dandb	r15, 16, r15
	dcsel	r13, 4, r16
	dcmpneq32	r14, 0, r17
	dsubi32	r13, r16, r14
	dsubi32	10, r13, r5
	dcsel	r14, 0, r14
	dmaxi32	r5, 0, r7
	dsubi32	26, r15, r5
	dshlb	r14, 7, r14
	dmini32	r5, 20, r5
	daddi32	[rp4], r14, r14
	dshlb	r15, 2, r6
	dsubi32	60, r5, r28
	daddi32	r14, r6, r29
	dshrlb	r11, r28, r14
	dcmplti32	28, r5, r6
	dcsel	r14, 0, r28
	dsubi32	28, r5, r14
	dcmplti32	r5, 28, r5
	dshrlb	r11, r14, r5
	daddi32	r13, 16, r14
	dcsel	r5, -1, r5
	dcmpeq32	r15, 0, r30
	dandb	r5, -16, r6
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcsel	r6, r5, r30
	dcmplt32	r14, 26, r2
	daddi32	[rp4 + 3], r7, r7
	dsubi32	26, r13, r5
	dcmplt32	25, r14, r6
	dmuli32	r7, r2, r7
	dsubi32	4, r16, r16
	dmin32	r5, 16, r8
	dshlb	r6, 2, r5
	dcmpneq32	r17, 0, r17
	dxorb	r5, 20, r17
	dcsel	r16, 4, r5
	daddi32	r7, r8, r7
	dsubi32	4, r5, r16
	dmin32	r7, r17, r7
	dcp	r28, pls.maskh, south
	daddi32	r7, r16, r17
	dcp	r30, pls.maskl, south
	dcp	r29, pls.addr, south
	dcp	r17, pls.count1, south
	dstcr	1, r6
	dsubi32	20, r7, r16
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r5, 0, :.LBB5_36
// %bb.2:                               //   in Loop: Header=BB5_1 Depth=1
	dstcr	1, r6
	djmpeqoff	r17, 0, :.LBB5_21
// %bb.3:                               //   in Loop: Header=BB5_1 Depth=1
	dstcr	1, r7
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	0, r16, :.LBB5_13
// %bb.4:                               //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB5_5:                                // %.preheader23
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_6 Depth 3
                                        //       Child Loop BB5_8 Depth 3
                                        //       Child Loop BB5_10 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB5_6
.LBB5_6:                                //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB5_5 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB5_8
.LBB5_8:                                //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB5_5 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB5_10
.LBB5_10:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB5_5 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, 32, :.LBB5_5
// %bb.12:                              // %Flow
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r7
.LBB5_13:                               // %Flow5
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	r7, 0, :.LBB5_20
// %bb.14:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB5_15:                               // %.preheader21
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_16 Depth 3
                                        //       Child Loop BB5_18 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB5_16
.LBB5_16:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB5_15 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB5_18
.LBB5_18:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB5_15 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, 32, :.LBB5_15
.LBB5_20:                               // %Flow6
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r6
.LBB5_21:                               // %Flow11
                                        //   in Loop: Header=BB5_1 Depth=1
	djmpeqoff	r6, 0, :.LBB5_35
// %bb.22:                              //   in Loop: Header=BB5_1 Depth=1
	dstcr	1, r7
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	0, r16, :.LBB5_30
// %bb.23:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB5_24:                               // %.preheader19
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_25 Depth 3
                                        //       Child Loop BB5_27 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB5_25
.LBB5_25:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB5_24 Depth=2
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB5_27
.LBB5_27:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB5_24 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, 32, :.LBB5_24
// %bb.29:                              // %Flow7
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r7
.LBB5_30:                               // %Flow9
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	r7, 0, :.LBB5_35
// %bb.31:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB5_32:                               // %.preheader17
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_33 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB5_33
.LBB5_33:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB5_32 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, 32, :.LBB5_32
.LBB5_35:                               // %Flow12
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r6
.LBB5_36:                               // %Flow23
                                        //   in Loop: Header=BB5_1 Depth=1
	djmpeqoff	r6, 0, :.LBB5_62
// %bb.37:                              //   in Loop: Header=BB5_1 Depth=1
	dstcr	1, r5
	djmpeqoff	r17, 0, :.LBB5_52
// %bb.38:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	1040, crp3
	dstcr	1, r6
	addi32	crp1, crp3, crp3
	dstcr	0, r5
	cp	crp3, crp4
	djmpeqoff	0, r16, :.LBB5_46
// %bb.39:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB5_40:                               // %.preheader15
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_41 Depth 3
                                        //       Child Loop BB5_43 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB5_41
.LBB5_41:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB5_40 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB5_43
.LBB5_43:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB5_40 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r5, 32, :.LBB5_40
// %bb.45:                              // %Flow13
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r6
.LBB5_46:                               // %Flow15
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r5
	djmpeqoff	r6, 0, :.LBB5_51
// %bb.47:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB5_48:                               // %.preheader13
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_49 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB5_49
.LBB5_49:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB5_48 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 32, :.LBB5_48
.LBB5_51:                               // %Flow16
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r5
.LBB5_52:                               // %Flow21
                                        //   in Loop: Header=BB5_1 Depth=1
	djmpeqoff	r5, 0, :.LBB5_62
// %bb.53:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	1040, crp3
	dstcr	1, r5
	addi32	crp1, crp3, crp3
	dstcr	0, r17
	cp	crp3, crp4
	djmpeqoff	0, r16, :.LBB5_59
// %bb.54:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB5_55:                               // %.preheader11
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_56 Depth 3
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB5_56
.LBB5_56:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB5_55 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r17, 32, :.LBB5_55
// %bb.58:                              // %Flow17
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0, r5
.LBB5_59:                               // %Flow19
                                        //   in Loop: Header=BB5_1 Depth=1
	djmpeqoff	r5, 0, :.LBB5_62
// %bb.60:                              //   in Loop: Header=BB5_1 Depth=1
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 32, :.LBB5_61
.LBB5_61:                               // %.preheader
                                        //   Parent Loop BB5_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp3+=1]
.LBB5_62:                               // %.loopexit
                                        //   in Loop: Header=BB5_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r17
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r17, pls.addr, west
	dshrlb	r15, 4, r15
	dstcr	0, r16
	addi32	crp1, 16, crp3          //      
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB5_63:                               //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_64 Depth 3
                                        //       Child Loop BB5_66 Depth 3
	cp	crp2, crp4
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 16, :.LBB5_64
	stcr	0x0, accumall
.LBB5_64:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrxi8.lb	[crp4+=2]
// %bb.65:                              //   in Loop: Header=BB5_63 Depth=2
	addi32	crp4, -128, crp4
	stcr	0x1, bitwidthmode
	djmpincsetup	0, 63, :.LBB5_66
	nrb	[crp4.z+=1], north | south | east | west
.LBB5_66:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp4.z+=1], north | south | east | west
// %bb.67:                              //   in Loop: Header=BB5_63 Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr14
	cp	accum0h, cr15
	stcr	0x2, bitwidthmode
	addi32	cr14, cr15, cr14
	muli32lohi{21}	cr14, cr10, cr14
	mini32	cr14, 127, cr14
	maxi32	cr14, -127, cr14
	muli32	>wl, cr14, cr14
	addi32	>wl, cr14, cr14
	muli32lohi{20}	cr14, cr11, cr14
	mini32	cr14, 127, cr14
	maxi32	cr14, -127, cr14
	shlb	cr14, 16, cr14
	muli32lohi{16}	cr14, cr12, cr14
	muli32lohi{16}	cr14, cr13, cr15
	cmplti32	0, cr14, cr16
	csel	cr14, cr15, [crp3+=1]
	djmpincne	r16, r12, :.LBB5_63
// %bb.68:                              //   in Loop: Header=BB5_1 Depth=1
	dshlb	r13, 7, r13
	dshlb	r15, 6, r15
	daddi32	[rp2], r13, r13
	dcmplt32	26, r14, r14
	daddi32	r13, r15, r15
	dcsel	10, 16, r13
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r15, pls.addr, north
	dcp	r13, pls.count1, north
	dstcr	0, r14
	addi32	crp1, 16, crp3          //      
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
.LBB5_69:                               //   Parent Loop BB5_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_70 Depth 3
                                        //       Child Loop BB5_72 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB5_70
	dstcr	0x200, pc.mode, north
.LBB5_70:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.71:                              //   in Loop: Header=BB5_69 Depth=2
	dcp	r13, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB5_72
.LBB5_72:                               //   Parent Loop BB5_1 Depth=1
                                        //     Parent Loop BB5_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB5_69 Depth=2
	addi32	crp3, 4, crp3
	djmpincne	r14, r12, :.LBB5_69
// %bb.74:                              //   in Loop: Header=BB5_1 Depth=1
	dstcr	0x200, pc.mode, north
	djmpincne	r10, 4, :.LBB5_1
// %bb.75:
	dcp	[rp1 + 4], r8
	daddi32	rp1, 24, rp1
	stcr	1168, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z61fused_nn_max_pool2d_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_
_Z61fused_nn_max_pool2d_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_: // @_Z61fused_nn_max_pool2d_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-1296, cr10
	stcr	272, crp2
	addi32	crp1, cr10, crp1        //     
	dcp	r11, rp2
	dcp	r10, rp3
	dstcr	0, r10
	dstcr	-1, r11
	addi32	crp1, crp2, crp2
	dstcr	256, r12
	stcr	14210, cr10
	stcr	-32768, cr11
	stcr	32768, cr12
	stcr	8323072, cr13
	stcr	-8323072, cr14
	dstcr	0, rp4
	dstcr	0x20, pls.stride1, south
	dstcr	0x100, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x340, pls.stride2, south
	dstcr	0x2, mode
	dstcr	0x0, plsthresholdnorth
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB6_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_5 Depth 2
                                        //       Child Loop BB6_6 Depth 3
                                        //       Child Loop BB6_8 Depth 3
                                        //       Child Loop BB6_10 Depth 3
                                        //     Child Loop BB6_15 Depth 2
                                        //       Child Loop BB6_16 Depth 3
                                        //       Child Loop BB6_18 Depth 3
                                        //     Child Loop BB6_24 Depth 2
                                        //       Child Loop BB6_25 Depth 3
                                        //       Child Loop BB6_27 Depth 3
                                        //     Child Loop BB6_32 Depth 2
                                        //       Child Loop BB6_33 Depth 3
                                        //     Child Loop BB6_40 Depth 2
                                        //       Child Loop BB6_41 Depth 3
                                        //       Child Loop BB6_43 Depth 3
                                        //     Child Loop BB6_48 Depth 2
                                        //       Child Loop BB6_49 Depth 3
                                        //     Child Loop BB6_55 Depth 2
                                        //       Child Loop BB6_56 Depth 3
                                        //     Child Loop BB6_61 Depth 2
                                        //     Child Loop BB6_63 Depth 2
                                        //     Child Loop BB6_65 Depth 2
                                        //       Child Loop BB6_68 Depth 3
                                        //       Child Loop BB6_70 Depth 3
                                        //       Child Loop BB6_72 Depth 3
                                        //       Child Loop BB6_74 Depth 3
	dshrlb	r10, 1, r14
	dandb	r10, 1, r13
	dshlb	r14, 4, r16
	dcmpeq32	r14, 0, r15
	dcsel	r16, 4, r5
	dshlb	r13, 4, r17
	dsubi32	r16, r5, r7
	dcmpneq32	r14, 0, r6
	dcsel	r7, 0, r7
	dsubi32	26, r17, r17
	dshlb	r7, 7, r7
	dmini32	r17, 20, r17
	daddi32	[rp3], r7, r7
	dshlb	r13, 6, r29
	dsubi32	60, r17, r30
	daddi32	r7, r29, r7
	dshrlb	r11, r30, r29
	dcmplti32	28, r17, r30
	dsubi32	28, r17, r30
	dcsel	r29, 0, r29
	dcmplti32	r17, 28, r17
	dsubi32	10, r16, r28
	dshrlb	r11, r30, r17
	dmaxi32	r28, 0, r28
	dcsel	r17, -1, r17
	dcmpeq32	r13, 0, r30
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dandb	r17, -16, r30
	daddi32	[rp3 + 3], r28, r28
	dsubi32	26, r16, r16
	dcsel	r30, r17, r30
	dshlb	r6, 2, r17
	dmuli32	r28, r15, r15
	dmin32	r16, 16, r16
	dsubi32	4, r5, r5
	dxorb	r17, 20, r2
	dcmpneq32	r6, 0, r17
	dcsel	r5, 4, r17
	daddi32	r15, r16, r15
	dsubi32	4, r17, r6
	dmin32	r15, r2, r15
	dcp	r29, pls.maskh, south
	daddi32	r15, r6, r16
	dcp	r30, pls.maskl, south
	dcp	r7, pls.addr, south
	dcp	r16, pls.count1, south
	dstcr	1, r5
	dsubi32	20, r15, r15
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	djmpeqoff	r17, 0, :.LBB6_36
// %bb.2:                               //   in Loop: Header=BB6_1 Depth=1
	dstcr	1, r5
	djmpeqoff	r16, 0, :.LBB6_21
// %bb.3:                               //   in Loop: Header=BB6_1 Depth=1
	dstcr	1, r6
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	0, r15, :.LBB6_13
// %bb.4:                               //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB6_5:                                // %.preheader46
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_6 Depth 3
                                        //       Child Loop BB6_8 Depth 3
                                        //       Child Loop BB6_10 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB6_6
.LBB6_6:                                //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB6_5 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB6_8
.LBB6_8:                                //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB6_5 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB6_10
.LBB6_10:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB6_5 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, r12, :.LBB6_5
// %bb.12:                              // %Flow4
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r6
.LBB6_13:                               // %Flow6
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	r6, 0, :.LBB6_20
// %bb.14:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB6_15:                               // %.preheader44
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_16 Depth 3
                                        //       Child Loop BB6_18 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB6_16
.LBB6_16:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB6_15 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB6_18
.LBB6_18:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB6_15 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, r12, :.LBB6_15
.LBB6_20:                               // %Flow7
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r5
.LBB6_21:                               // %Flow12
                                        //   in Loop: Header=BB6_1 Depth=1
	djmpeqoff	r5, 0, :.LBB6_35
// %bb.22:                              //   in Loop: Header=BB6_1 Depth=1
	dstcr	1, r6
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	0, r15, :.LBB6_30
// %bb.23:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB6_24:                               // %.preheader42
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_25 Depth 3
                                        //       Child Loop BB6_27 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB6_25
.LBB6_25:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB6_24 Depth=2
	dcp	r15, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB6_27
.LBB6_27:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB6_24 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, r12, :.LBB6_24
// %bb.29:                              // %Flow8
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r6
.LBB6_30:                               // %Flow10
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	r6, 0, :.LBB6_35
// %bb.31:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB6_32:                               // %.preheader40
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_33 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB6_33
.LBB6_33:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB6_32 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, r12, :.LBB6_32
.LBB6_35:                               // %Flow13
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r5
.LBB6_36:                               // %Flow24
                                        //   in Loop: Header=BB6_1 Depth=1
	djmpeqoff	r5, 0, :.LBB6_62
// %bb.37:                              //   in Loop: Header=BB6_1 Depth=1
	dstcr	1, r17
	djmpeqoff	r16, 0, :.LBB6_52
// %bb.38:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	272, crp3
	dstcr	1, r5
	addi32	crp1, crp3, crp3
	dstcr	0, r17
	cp	crp3, crp4
	djmpeqoff	0, r15, :.LBB6_46
// %bb.39:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB6_40:                               // %.preheader38
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_41 Depth 3
                                        //       Child Loop BB6_43 Depth 3
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB6_41
.LBB6_41:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB6_40 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB6_43
.LBB6_43:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB6_40 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r17, r12, :.LBB6_40
// %bb.45:                              // %Flow14
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r5
.LBB6_46:                               // %Flow16
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r17
	djmpeqoff	r5, 0, :.LBB6_51
// %bb.47:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB6_48:                               // %.preheader36
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_49 Depth 3
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB6_49
.LBB6_49:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB6_48 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r17, r12, :.LBB6_48
.LBB6_51:                               // %Flow17
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r17
.LBB6_52:                               // %Flow22
                                        //   in Loop: Header=BB6_1 Depth=1
	djmpeqoff	r17, 0, :.LBB6_62
// %bb.53:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	272, crp3
	dstcr	1, r17
	addi32	crp1, crp3, crp3
	dstcr	0, r16
	cp	crp3, crp4
	djmpeqoff	0, r15, :.LBB6_59
// %bb.54:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB6_55:                               // %.preheader34
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_56 Depth 3
	dcp	r15, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB6_56
.LBB6_56:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB6_55 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r16, r12, :.LBB6_55
// %bb.58:                              // %Flow18
                                        //   in Loop: Header=BB6_1 Depth=1
	dstcr	0, r17
.LBB6_59:                               // %Flow20
                                        //   in Loop: Header=BB6_1 Depth=1
	djmpeqoff	r17, 0, :.LBB6_62
// %bb.60:                              //   in Loop: Header=BB6_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x100, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB6_61
.LBB6_61:                               // %.preheader
                                        //   Parent Loop BB6_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp3+=1]
.LBB6_62:                               // %.loopexit
                                        //   in Loop: Header=BB6_1 Depth=1
	stcr	272, crp3
	addi32	crp1, 16, crp4          //      
	addi32	crp1, crp3, crp3
	dstcr	0x200, pc.mode, south
	dstcr	0x100, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB6_63
.LBB6_63:                               //   Parent Loop BB6_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	stcr	0x2, bitwidthmode
	cp	[crp3+=1], cr15
	nrb	cr15, north
	maxi32	south, cr15, cr15
	nrb	cr15, west
	maxi32	east, cr15, cr15
	stcr	0x0, bitwidthmode
	muli32lohi{10}	cr15, cr10, cr15
	cmplti32	0, cr15, cr16
	csel	cr12, cr11, cr16
	addi32	cr16, cr15, cr15
	mini32	cr15, cr13, cr15
	maxi32	cr15, cr14, cr15
	shrlb.lb	cr15, 16, [crp4.z+=1]
// %bb.64:                              //   in Loop: Header=BB6_1 Depth=1
	dcp	[rp2 + 1], r15
	cp	row, cr15
	cp	col, cr16
	shrlb	cr15, 31, cr17
	dshlb	r14, 3, r14
	addi32	cr15, cr17, cr17
	shrlb	cr16, 31, cr5
	shrab	cr17, 1, cr17
	daddi32	r14, 8, r16
	addi32	cr16, cr5, cr5
	dshlb	r13, 3, r13
	orb	cr16, cr15, cr16
	dcpc	r14, cr15
	addi32	cr17, cr15, cr15
	shrab	cr5, 1, cr5
	daddi32	r13, 8, r14
	dcpc	r16, cr17
	cmpltei32	cr17, cr15, cr17
	cmplti32	12, cr15, cr6
	dcpc	r13, cr7
	addi32	cr5, cr7, cr5
	orb	cr6, cr17, cr17
	dcpc	r14, cr6
	cmpltei32	cr6, cr5, cr6
	cmplti32	12, cr5, cr7
	shlb	cr15, 4, cr15
	orb	cr7, cr6, cr6
	addi32	cr15, cr5, cr15
	orb	cr17, cr6, cr17
	andb	cr16, 1, cr16
	xorb	cr17, 1, cr17
	dcp	rp4, r13
	dcp	[rp2], pls.addr, north
	dcpc	rp4, crp3
	stcr	0x2, bitwidthmode
	dstcr	0x1, pc.resetfifo, north
.LBB6_65:                               //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_68 Depth 3
                                        //       Child Loop BB6_70 Depth 3
                                        //       Child Loop BB6_72 Depth 3
                                        //       Child Loop BB6_74 Depth 3
	cmpeq32	cr16, 0, cr5
	predpush	cr5, :.LBB6_77
// %bb.66:                              //   in Loop: Header=BB6_65 Depth=2
	predpush	cr17, :.LBB6_76
// %bb.67:                              //   in Loop: Header=BB6_65 Depth=2
	dmuli32	r13, 208, r14
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcpc	r14, cr5
	dcp	r15, dependencyid
	addi32	cr15, cr5, cr5
	dstcr	0x1, plsstatus, north
	dcp	flowid, r15
	shlb	cr5, 2, cr5
	djmpincsetup	0, 4, :.LBB6_68
	dstcr	0x260, pc.mode, north
	nrb	cr5, north
.LBB6_68:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.69:                              //   in Loop: Header=BB6_65 Depth=2
	djmpincsetup	0, 16, :.LBB6_70
	dstcr	0x360, pc.mode, north
.LBB6_70:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.71:                              //   in Loop: Header=BB6_65 Depth=2
	shlb	crp3, 2, crp4
	addi32	crp1, 16, crp5          //      
	dstcr	0x260, pc.mode, north
	addi32	crp5, crp4, crp4
	djmpincsetup	0, 4, :.LBB6_72
	nrb	[crp4], north
.LBB6_72:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB6_65 Depth=2
	djmpincsetup	0, 16, :.LBB6_74
	dstcr	0x360, pc.mode, north
.LBB6_74:                               //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.75:                              //   in Loop: Header=BB6_65 Depth=2
	dstcr	0x260, pc.mode, north
.LBB6_76:                               // %Flow
                                        //   in Loop: Header=BB6_65 Depth=2
	predpop	
	addi32	crp3, 1, crp3
.LBB6_77:                               // %Flow3
                                        //   in Loop: Header=BB6_65 Depth=2
	predpop	
	djmpincne	r13, 64, :.LBB6_65
// %bb.78:                              //   in Loop: Header=BB6_1 Depth=1
	dstcr	-261, r31
	djmpincne	r10, 4, r31
.LBB6_79:
	daddi32	rp1, 16, rp1
	stcr	1296, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_: // @_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_7707757238766254228_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-528, cr10
	dcp	r10, rp4
	addi32	crp1, cr10, crp1        //     
	dstcr	0x2, mode
	stcr	272, crp2
	dcp	r11, rp3
	dcp	[rp4], r11
	addi32	crp1, crp2, crp2
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dstcr	0x0, pls.maskh, south
	dstcr	0x1fff0, pls.maskl, south
	dcp	r11, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0xd, pls.count1, south
	dstcr	0x40, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xd0, pls.stride2, south
	dcp	r12, rp2
	dstcr	0, r10
	cp	crp2, crp3
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	stcr	0x2, bitwidthmode
.LBB7_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_2 Depth 2
                                        //     Child Loop BB7_4 Depth 2
                                        //     Child Loop BB7_6 Depth 2
	djmpincsetup	0, 4, :.LBB7_2
	dstcr	0x200, pc.mode, south
.LBB7_2:                                //   Parent Loop BB7_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB7_1 Depth=1
	djmpincsetup	0, 13, :.LBB7_4
	dstcr	0x300, pc.mode, south
.LBB7_4:                                //   Parent Loop BB7_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB7_1 Depth=1
	djmpincsetup	0, 7, :.LBB7_6
	dstcr	0x200, pc.mode, south
.LBB7_6:                                //   Parent Loop BB7_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB7_1 Depth=1
	cp	south, [crp3+=1]
	djmpincne	r10, 64, :.LBB7_1
// %bb.8:
	dstcr	0x200, pc.mode, south
	stcr	0x0, vapmode
	dcp	[rp3], r12
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dcp	r12, pls.addr, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x12200, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0, r10
	addi32	crp1, 16, crp3          //      
	stcr	12492, cr10
	stcr	12243, cr11
	stcr	15245, cr12
	stcr	6553, cr13
	stcr	10726, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	256, r11
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB7_9:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_10 Depth 2
                                        //     Child Loop BB7_12 Depth 2
	cp	crp2, crp4
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 32, :.LBB7_10
	stcr	0x0, accumall
.LBB7_10:                               //   Parent Loop BB7_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrxi8.lb	[crp4+=2]
// %bb.11:                              //   in Loop: Header=BB7_9 Depth=1
	addi32	crp4, -256, crp4
	stcr	0x1, bitwidthmode
	djmpincsetup	0, 127, :.LBB7_12
	nrb	[crp4.z+=1], north | south | east | west
.LBB7_12:                               //   Parent Loop BB7_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp4.z+=1], north | south | east | west
// %bb.13:                              //   in Loop: Header=BB7_9 Depth=1
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{22}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	muli32lohi{20}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{17}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	nrb	cr6, north
	maxi32	south, cr6, cr6
	nrb	cr6, west
	maxi32	east, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{9}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp3.z+=1]
	djmpincne	r10, r11, :.LBB7_9
// %bb.14:
	dcp	[rp2], r11
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dcp	r11, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xd, pls.count1, north
	dstcr	0x40, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xd0, pls.stride2, north
	addi32	crp1, 16, crp2          //      
	dstcr	0, r10
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	stcr	0x2, bitwidthmode
.LBB7_15:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_16 Depth 2
                                        //     Child Loop BB7_18 Depth 2
	nrb	[crp2], north
	djmpincsetup	0, 4, :.LBB7_16
	dstcr	0x200, pc.mode, north
.LBB7_16:                               //   Parent Loop BB7_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB7_15 Depth=1
	djmpincsetup	0, 13, :.LBB7_18
	dstcr	0x300, pc.mode, north
.LBB7_18:                               //   Parent Loop BB7_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB7_15 Depth=1
	addi32	crp2, 4, crp2
	djmpincne	r10, 64, :.LBB7_15
// %bb.20:
	dstcr	0x200, pc.mode, north
	daddi32	rp1, 16, rp1
	stcr	528, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj2367488EEES6_EvRT0_RT1_RT2_
_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj2367488EEES6_EvRT0_RT1_RT2_: // @_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__2I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj2367488EEES6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-1040, cr10
	dcp	r10, rp4
	addi32	crp1, cr10, crp1        //     
	dstcr	0x2, mode
	stcr	528, crp2
	dcp	r11, rp3
	dcp	[rp4], r11
	addi32	crp1, crp2, crp2
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dstcr	0x0, pls.maskh, south
	dstcr	0x1fff0, pls.maskl, south
	dcp	r11, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0xd, pls.count1, south
	dstcr	0x80, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xd0, pls.stride2, south
	dcp	r12, rp2
	dstcr	0, r10
	cp	crp2, crp3
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	stcr	0x2, bitwidthmode
.LBB8_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_2 Depth 2
                                        //     Child Loop BB8_4 Depth 2
                                        //     Child Loop BB8_6 Depth 2
	djmpincsetup	0, 4, :.LBB8_2
	dstcr	0x200, pc.mode, south
.LBB8_2:                                //   Parent Loop BB8_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB8_1 Depth=1
	djmpincsetup	0, 13, :.LBB8_4
	dstcr	0x300, pc.mode, south
.LBB8_4:                                //   Parent Loop BB8_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB8_1 Depth=1
	djmpincsetup	0, 7, :.LBB8_6
	dstcr	0x200, pc.mode, south
.LBB8_6:                                //   Parent Loop BB8_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB8_1 Depth=1
	cp	south, [crp3+=1]
	djmpincne	r10, 128, :.LBB8_1
// %bb.8:
	dstcr	0x200, pc.mode, south
	stcr	0x0, vapmode
	dcp	[rp3], r12
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dcp	r12, pls.addr, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x48400, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0, r10
	addi32	crp1, 16, crp3          //      
	stcr	-512, crp4
	stcr	8449, cr10
	stcr	11788, cr11
	stcr	12141, cr12
	stcr	6553, cr13
	stcr	13411, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	512, r11
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
	dstcr	0xff, jumpendcount
.LBB8_9:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_10 Depth 2
                                        //     Child Loop BB8_12 Depth 2
	cp	crp2, crp5
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 64, :.LBB8_10
	stcr	0x0, accumall
.LBB8_10:                               //   Parent Loop BB8_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrxi8.lb	[crp5+=2]
// %bb.11:                              //   in Loop: Header=BB8_9 Depth=1
	addi32	crp5, crp4, crp5
	stcr	0x1, bitwidthmode
	nrb	[crp5.z+=1], north | south | east | west
	djmpincsetup	0, jumpendcount, :.LBB8_12
.LBB8_12:                               //   Parent Loop BB8_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp5.z+=1], north | south | east | west
// %bb.13:                              //   in Loop: Header=BB8_9 Depth=1
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{22}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{20}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{15}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	muli32lohi{11}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp3.z+=1]
	djmpincne	r10, r11, :.LBB8_9
// %bb.14:
	dcp	[rp2], r11
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dcp	r11, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xd, pls.count1, north
	dstcr	0x80, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xd0, pls.stride2, north
	addi32	crp1, 16, crp2          //      
	dstcr	0, r10
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	stcr	0x2, bitwidthmode
.LBB8_15:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_16 Depth 2
                                        //     Child Loop BB8_18 Depth 2
	nrb	[crp2], north
	djmpincsetup	0, 4, :.LBB8_16
	dstcr	0x200, pc.mode, north
.LBB8_16:                               //   Parent Loop BB8_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB8_15 Depth=1
	djmpincsetup	0, 13, :.LBB8_18
	dstcr	0x300, pc.mode, north
.LBB8_18:                               //   Parent Loop BB8_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB8_15 Depth=1
	addi32	crp2, 4, crp2
	djmpincne	r10, 128, :.LBB8_15
// %bb.20:
	dstcr	0x200, pc.mode, north
	daddi32	rp1, 16, rp1
	stcr	1040, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj266240EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_RT2_
_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj266240EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_RT2_: // @_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj256ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj266240EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj13ELj13EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-1296, cr10
	dcp	r10, rp4
	addi32	crp1, cr10, crp1        //     
	dstcr	0x2, mode
	stcr	272, crp2
	dcp	r12, rp2
	dcp	[rp4], r12
	addi32	crp1, crp2, crp2
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dstcr	0x0, pls.maskh, south
	dstcr	0x1fff0, pls.maskl, south
	dcp	r12, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0xd, pls.count1, south
	dstcr	0x100, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xd0, pls.stride2, south
	dcp	r11, rp3
	dstcr	0, r10
	dstcr	256, r11
	cp	crp2, crp3
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	stcr	0x2, bitwidthmode
.LBB9_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_2 Depth 2
                                        //     Child Loop BB9_4 Depth 2
                                        //     Child Loop BB9_6 Depth 2
	djmpincsetup	0, 4, :.LBB9_2
	dstcr	0x200, pc.mode, south
.LBB9_2:                                //   Parent Loop BB9_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB9_1 Depth=1
	djmpincsetup	0, 13, :.LBB9_4
	dstcr	0x300, pc.mode, south
.LBB9_4:                                //   Parent Loop BB9_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB9_1 Depth=1
	djmpincsetup	0, 7, :.LBB9_6
	dstcr	0x200, pc.mode, south
.LBB9_6:                                //   Parent Loop BB9_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB9_1 Depth=1
	cp	south, [crp3+=1]
	djmpincne	r10, r11, :.LBB9_1
// %bb.8:
	dstcr	0x200, pc.mode, south
	stcr	0x0, vapmode
	dcp	[rp3], r12
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dcp	r12, pls.addr, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x8200, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0, r10
	addi32	crp1, 16, crp3          //      
	stcr	8811, cr10
	stcr	8274, cr11
	stcr	14921, cr12
	stcr	6553, cr13
	stcr	8234, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	256, r11
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB9_9:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_10 Depth 2
	cp	crp2, crp4
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 128, :.LBB9_10
	stcr	0x0, accumall
.LBB9_10:                               //   Parent Loop BB9_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrxi8.lb	[crp4+=2]
// %bb.11:                              //   in Loop: Header=BB9_9 Depth=1
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	addi32	cr6, cr7, cr6
	muli32lohi{21}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{19}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{18}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	muli32lohi{8}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp3.z+=1]
	djmpincne	r10, r11, :.LBB9_9
// %bb.12:
	dcp	[rp2], r11
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dcp	r11, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xd, pls.count1, north
	dstcr	0x40, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xd0, pls.stride2, north
	addi32	crp1, 16, crp2          //      
	dstcr	0, r10
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	stcr	0x2, bitwidthmode
.LBB9_13:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_14 Depth 2
                                        //     Child Loop BB9_16 Depth 2
	nrb	[crp2], north
	djmpincsetup	0, 4, :.LBB9_14
	dstcr	0x200, pc.mode, north
.LBB9_14:                               //   Parent Loop BB9_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.15:                              //   in Loop: Header=BB9_13 Depth=1
	djmpincsetup	0, 13, :.LBB9_16
	dstcr	0x300, pc.mode, north
.LBB9_16:                               //   Parent Loop BB9_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB9_13 Depth=1
	addi32	crp2, 4, crp2
	djmpincne	r10, 64, :.LBB9_13
// %bb.18:
	dstcr	0x200, pc.mode, north
	daddi32	rp1, 16, rp1
	stcr	1296, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_: // @_Z101fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj593920EEES6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-528, cr10
	dcp	r10, rp4
	addi32	crp1, cr10, crp1        //     
	dstcr	0x2, mode
	stcr	272, crp2
	dcp	r11, rp3
	dcp	[rp4], r11
	addi32	crp1, crp2, crp2
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dstcr	0x0, pls.maskh, south
	dstcr	0x1fff0, pls.maskl, south
	dcp	r11, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0xd, pls.count1, south
	dstcr	0x40, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xd0, pls.stride2, south
	dcp	r12, rp2
	dstcr	0, r10
	cp	crp2, crp3
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	stcr	0x2, bitwidthmode
.LBB10_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_2 Depth 2
                                        //     Child Loop BB10_4 Depth 2
                                        //     Child Loop BB10_6 Depth 2
	djmpincsetup	0, 4, :.LBB10_2
	dstcr	0x200, pc.mode, south
.LBB10_2:                               //   Parent Loop BB10_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB10_1 Depth=1
	djmpincsetup	0, 13, :.LBB10_4
	dstcr	0x300, pc.mode, south
.LBB10_4:                               //   Parent Loop BB10_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB10_1 Depth=1
	djmpincsetup	0, 7, :.LBB10_6
	dstcr	0x200, pc.mode, south
.LBB10_6:                               //   Parent Loop BB10_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB10_1 Depth=1
	cp	south, [crp3+=1]
	djmpincne	r10, 64, :.LBB10_1
// %bb.8:
	dstcr	0x200, pc.mode, south
	stcr	0x0, vapmode
	dcp	[rp3], r12
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dcp	r12, pls.addr, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x12200, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0, r10
	addi32	crp1, 16, crp3          //      
	stcr	13748, cr10
	stcr	15700, cr11
	stcr	15918, cr12
	stcr	6553, cr13
	stcr	8524, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	256, r11
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB10_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_10 Depth 2
                                        //     Child Loop BB10_12 Depth 2
	cp	crp2, crp4
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 32, :.LBB10_10
	stcr	0x0, accumall
.LBB10_10:                              //   Parent Loop BB10_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrxi8.lb	[crp4+=2]
// %bb.11:                              //   in Loop: Header=BB10_9 Depth=1
	addi32	crp4, -256, crp4
	stcr	0x1, bitwidthmode
	djmpincsetup	0, 127, :.LBB10_12
	nrb	[crp4.z+=1], north | south | east | west
.LBB10_12:                              //   Parent Loop BB10_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp4.z+=1], north | south | east | west
// %bb.13:                              //   in Loop: Header=BB10_9 Depth=1
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{23}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{20}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{17}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	muli32lohi{9}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp3.z+=1]
	djmpincne	r10, r11, :.LBB10_9
// %bb.14:
	dcp	[rp2], r11
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dcp	r11, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xd, pls.count1, north
	dstcr	0x40, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xd0, pls.stride2, north
	addi32	crp1, 16, crp2          //      
	dstcr	0, r10
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	stcr	0x2, bitwidthmode
.LBB10_15:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_16 Depth 2
                                        //     Child Loop BB10_18 Depth 2
	nrb	[crp2], north
	djmpincsetup	0, 4, :.LBB10_16
	dstcr	0x200, pc.mode, north
.LBB10_16:                              //   Parent Loop BB10_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB10_15 Depth=1
	djmpincsetup	0, 13, :.LBB10_18
	dstcr	0x300, pc.mode, north
.LBB10_18:                              //   Parent Loop BB10_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB10_15 Depth=1
	addi32	crp2, 4, crp2
	djmpincne	r10, 64, :.LBB10_15
// %bb.20:
	dstcr	0x200, pc.mode, north
	daddi32	rp1, 16, rp1
	stcr	528, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z102fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj132600EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj13ELj13EEEEvRT0_RT1_RT2_
_Z102fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj132600EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj13ELj13EEEEvRT0_RT1_RT2_: // @_Z102fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj132600EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj13ELj13EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-1552, cr10
	dcp	r10, rp4
	addi32	crp1, cr10, crp1        //     
	dstcr	0x2, mode
	stcr	1040, crp2
	dcp	r11, rp3
	dcp	[rp4], r11
	addi32	crp1, crp2, crp2
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dstcr	0x0, pls.maskh, south
	dstcr	0x1fff0, pls.maskl, south
	dcp	r11, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0xd, pls.count1, south
	dstcr	0x80, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xd0, pls.stride2, south
	dcp	r12, rp2
	dstcr	0, r10
	cp	crp2, crp3
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	stcr	0x2, bitwidthmode
.LBB11_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_2 Depth 2
                                        //     Child Loop BB11_4 Depth 2
                                        //     Child Loop BB11_6 Depth 2
	djmpincsetup	0, 4, :.LBB11_2
	dstcr	0x200, pc.mode, south
.LBB11_2:                               //   Parent Loop BB11_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB11_1 Depth=1
	djmpincsetup	0, 13, :.LBB11_4
	dstcr	0x300, pc.mode, south
.LBB11_4:                               //   Parent Loop BB11_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB11_1 Depth=1
	djmpincsetup	0, 7, :.LBB11_6
	dstcr	0x200, pc.mode, south
.LBB11_6:                               //   Parent Loop BB11_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB11_1 Depth=1
	cp	south, [crp3+=1]
	djmpincne	r10, 128, :.LBB11_1
// %bb.8:
	dstcr	0x200, pc.mode, south
	stcr	0x0, vapmode
	dcp	[rp3], r11
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dcp	r11, pls.addr, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x40bf, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0, r10
	addi32	crp1, 16, crp3          //      
	stcr	9648, cr10
	stcr	13973, cr11
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB11_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_10 Depth 2
	cp	crp2, crp4
	djmpincsetup	0, 64, :.LBB11_10
	stcr	0x0, accumall
.LBB11_10:                              //   Parent Loop BB11_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrxi8.lb	[crp4+=2]
// %bb.11:                              //   in Loop: Header=BB11_9 Depth=1
	accsumsh8	0
	cp	accum0, cr12
	cp	accum0h, cr13
	addi32	cr12, cr13, cr12
	addi32	>wl, cr12, cr12
	muli32lohi{22}	cr12, cr10, cr12
	mini32	cr12, 127, cr12
	maxi32	cr12, -127, cr12
	shlb	cr12, 16, cr12
	muli32lohi{16}	cr12, cr11, [crp3+=1]
	djmpincne	r10, 255, :.LBB11_9
// %bb.12:
	dcp	[rp2], r11
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dcp	r11, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xd, pls.count1, north
	dstcr	0xff, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xd0, pls.stride2, north
	addi32	crp1, 16, crp2          //      
	dstcr	0, r10
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
.LBB11_13:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_14 Depth 2
                                        //     Child Loop BB11_16 Depth 2
	nrb	[crp2], north
	djmpincsetup	0, 4, :.LBB11_14
	dstcr	0x200, pc.mode, north
.LBB11_14:                              //   Parent Loop BB11_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.15:                              //   in Loop: Header=BB11_13 Depth=1
	djmpincsetup	0, 13, :.LBB11_16
	dstcr	0x300, pc.mode, north
.LBB11_16:                              //   Parent Loop BB11_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB11_13 Depth=1
	addi32	crp2, 4, crp2
	djmpincne	r10, 255, :.LBB11_13
// %bb.18:
	dstcr	0x200, pc.mode, north
	daddi32	rp1, 16, rp1
	stcr	1552, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z16fused_reshape_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj13ELj13EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj169EEEEvRT0_RT1_
_Z16fused_reshape_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj13ELj13EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj169EEEEvRT0_RT1_: // @_Z16fused_reshape_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj13ELj13EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj169EEEEvRT0_RT1_
// %bb.0:
	shlb	row, 4, cr10
	dcp	r11, rp2
	addi32	cr10, col, cr10
	dcp	r10, rp3
	dstcr	0, r13
	stcr	1626496491, cr11
	cmplti32	cr10, 169, cr12
	dstcr	33686018, r10
	stcr	1321528399, cr13
	dstcr	704, r11
                                        // implicit-def: $cx14
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	dstcr	0x2, mode
	stcr	0x2, bitwidthmode
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xb0, pls.stride2, north
.LBB12_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB12_3 Depth 2
                                        //     Child Loop BB12_5 Depth 2
                                        //     Child Loop BB12_7 Depth 2
                                        //     Child Loop BB12_9 Depth 2
                                        //     Child Loop BB12_11 Depth 2
                                        //     Child Loop BB12_16 Depth 2
                                        //     Child Loop BB12_18 Depth 2
	dmuli32	r13, 169, r14
	dcpc	r13, cr16
	daddi32	r13, 1, r12
	dcp	[rp3], pls.addr, north
	dcpc	r14, cr15
	addi32	cr15, cr10, cr17
	dstcr	0x1, pc.resetfifo, north
	muli32hi	cr17, cr11, cr15
	dstcr	0x10, pls.count1, north
	shrlb	cr15, 31, cr5
	shrab	cr15, 6, cr15
	addi32	cr15, cr5, cr15
	muli32	cr15, -169, cr5
	cmpltei32	cr15, cr16, cr6
	addi32	cr5, cr17, cr16
	andb	cr12, cr6, cr17
	predpush	cr17, :.LBB12_13
// %bb.2:                               //   in Loop: Header=BB12_1 Depth=1
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	[rp3 + 1], dependencyid
.LBB12_3:                               //   Parent Loop BB12_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dandb	r10, plsstatus, r14
	dorb	r14, pelsr, r14
	djmpneqoff	r14, 0, :.LBB12_3
// %bb.4:                               //   in Loop: Header=BB12_1 Depth=1
	muli32hi	cr16, cr13, cr14
	muli32	cr15, 208, cr15
	shrlb	cr14, 31, cr17
	shrlb	cr14, 2, cr14
	addi32	cr16, cr15, cr15
	addi32	cr14, cr17, cr14
	djmpincsetup	0, 4, :.LBB12_5
	muli32	cr14, 3, cr14
	dstcr	0x1, plsstatus, north
	addi32	cr15, cr14, cr14
	shlb	cr14, 2, cr14
	nrb	cr14, north
	dstcr	0x260, pc.mode, north
.LBB12_5:                               //   Parent Loop BB12_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB12_1 Depth=1
	djmpincsetup	0, 16, :.LBB12_7
	dstcr	0x360, pc.mode, north
.LBB12_7:                               //   Parent Loop BB12_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB12_1 Depth=1
	djmpincsetup	0, 16, :.LBB12_9
.LBB12_9:                               // %.preheader
                                        //   Parent Loop BB12_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.10:                              //   in Loop: Header=BB12_1 Depth=1
	djmpincsetup	0, 4, :.LBB12_11
	dstcr	0x260, pc.mode, north
.LBB12_11:                              //   Parent Loop BB12_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.12:                              //   in Loop: Header=BB12_1 Depth=1
	cp	north, cr14
.LBB12_13:                              // %Flow
                                        //   in Loop: Header=BB12_1 Depth=1
	predpop	
	dmuli32	r13, r11, r13
	daddi32	[rp2], r13, r13
	shlb	row, 4, cr15
	addi32	cr15, col, cr15
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r13, pls.addr, north
	dstcr	0xb, pls.count1, north
	cmplti32	cr15, 169, cr15
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	predpush	cr15, :.LBB12_15
// %bb.14:                              //   in Loop: Header=BB12_1 Depth=1
	nrb	cr14, north
.LBB12_15:                              //   in Loop: Header=BB12_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB12_16
	dstcr	0x200, pc.mode, north
.LBB12_16:                              //   Parent Loop BB12_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB12_1 Depth=1
	djmpincsetup	0, 11, :.LBB12_18
	dstcr	0x300, pc.mode, north
.LBB12_18:                              //   Parent Loop BB12_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB12_1 Depth=1
	dcp	r12, r13
	dstcr	0x200, pc.mode, north
	djmpneqoff	r12, 255, :.LBB12_1
// %bb.20:
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z18fused_transpose_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj169EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj169ELj1ELj255EEEEvRT0_RT1_
_Z18fused_transpose_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj169EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj169ELj1ELj255EEEEvRT0_RT1_: // @_Z18fused_transpose_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj169EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj169ELj1ELj255EEEEvRT0_RT1_
// %bb.0:
	addi32	crp1, -64, crp1         //     
	cp	row, cr10
	dcp	r11, rp2
	dcp	r10, rp3
	cp	col, cr11
	dstcr	0, r10
	cp	crp1, crp2
	dstcr	268435440, r11
	cmpeq32	cr10, 0, cr12
	dstcr	33686018, r12
	dstcr	2703, r13
	dstcr	0, r14
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	dstcr	0x2, mode
	stcr	0x2, bitwidthmode
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x100, pls.stride1, north
	dstcr	0x10, pls.stride2, north
.LBB13_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB13_2 Depth 2
                                        //       Child Loop BB13_4 Depth 3
                                        //       Child Loop BB13_6 Depth 3
                                        //       Child Loop BB13_8 Depth 3
                                        //       Child Loop BB13_10 Depth 3
                                        //       Child Loop BB13_12 Depth 3
                                        //     Child Loop BB13_16 Depth 2
                                        //       Child Loop BB13_17 Depth 3
	cp	crp2, crp3
	dstcr	0, r15
	dcp	[rp3], pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB13_2:                               //   Parent Loop BB13_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB13_4 Depth 3
                                        //       Child Loop BB13_6 Depth 3
                                        //       Child Loop BB13_8 Depth 3
                                        //       Child Loop BB13_10 Depth 3
                                        //       Child Loop BB13_12 Depth 3
	dshrab	r10, 31, r16
	stcr	0, cr13
	dshrlb	r16, 28, r16
	daddi32	r10, r16, r16
	dshrab	r16, 4, r17
	dandb	r16, r11, r16
	dcmplt32	r17, 169, r5
	dsubi32	r10, r16, r16
	dcpc	r17, cr14
	dshlb	r16, 4, r16
	dcpc	r5, cr15
	dcpc	r16, cr16
	addi32	cr16, cr11, cr16
	addi32	cr16, cr10, cr17
	cmplt32	cr16, 255, cr16
	muli32	cr17, 176, cr17
	andb	cr12, cr16, cr16
	addi32	cr17, cr14, cr14
	andb	cr15, cr16, cr15
	predpush	cr15, :.LBB13_14
// %bb.3:                               //   in Loop: Header=BB13_2 Depth=2
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	[rp3 + 1], dependencyid
.LBB13_4:                               //   Parent Loop BB13_1 Depth=1
                                        //     Parent Loop BB13_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	dandb	r12, plsstatus, r16
	dorb	r16, pelsr, r16
	djmpneqoff	r16, 0, :.LBB13_4
// %bb.5:                               //   in Loop: Header=BB13_2 Depth=2
	shlb	cr14, 2, cr13
	djmpincsetup	0, 4, :.LBB13_6
	dstcr	0x1, plsstatus, north
	nrb	cr13, north
	dstcr	0x260, pc.mode, north
.LBB13_6:                               //   Parent Loop BB13_1 Depth=1
                                        //     Parent Loop BB13_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB13_2 Depth=2
	djmpincsetup	0, 16, :.LBB13_8
	dstcr	0x360, pc.mode, north
.LBB13_8:                               //   Parent Loop BB13_1 Depth=1
                                        //     Parent Loop BB13_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB13_2 Depth=2
	djmpincsetup	0, 16, :.LBB13_10
.LBB13_10:                              // %.preheader
                                        //   Parent Loop BB13_1 Depth=1
                                        //     Parent Loop BB13_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.11:                              //   in Loop: Header=BB13_2 Depth=2
	djmpincsetup	0, 4, :.LBB13_12
	dstcr	0x260, pc.mode, north
.LBB13_12:                              //   Parent Loop BB13_1 Depth=1
                                        //     Parent Loop BB13_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.13:                              //   in Loop: Header=BB13_2 Depth=2
	cp	north, cr13
.LBB13_14:                              // %Flow
                                        //   in Loop: Header=BB13_2 Depth=2
	predpop	
	daddi32	r10, 1, r10
	cp	cr13, [crp3+=1]
	djmpincne	r15, 16, :.LBB13_2
// %bb.15:                              //   in Loop: Header=BB13_1 Depth=1
	dshlb	r14, 6, r15
	cp	crp2, crp3
	daddi32	[rp2], r15, r17
	dstcr	0, r15
	dcp	[rp2 + 1], r16
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r17, pls.addr, north
	dstcr	0x1, pls.count1, north
	dstcr	0x10, pls.count2, north
	dcp	r16, dependencyid
	dstcr	0x1, plsstatus, north
.LBB13_16:                              //   Parent Loop BB13_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB13_17 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB13_17
	dstcr	0x200, pc.mode, north
.LBB13_17:                              //   Parent Loop BB13_1 Depth=1
                                        //     Parent Loop BB13_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB13_16 Depth=2
	addi32	crp3, 4, crp3
	dstcr	0x300, pc.mode, north
	nnb	south, north
	djmpincne	r15, 16, :.LBB13_16
// %bb.19:                              //   in Loop: Header=BB13_1 Depth=1
	daddi32	r14, 16, r14
	dstcr	0x200, pc.mode, north
	djmplte	r14, r13, :.LBB13_1
// %bb.20:
	addi32	crp1, 64, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z16fused_reshape_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj169ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj507ELj1ELj85EEEEvRT0_RT1_
_Z16fused_reshape_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj169ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj507ELj1ELj85EEEEvRT0_RT1_: // @_Z16fused_reshape_13I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj169ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj507ELj1ELj85EEEEvRT0_RT1_
// %bb.0:
	dcp	r11, rp2
	dstcr	0x2, mode
	dcp	r10, rp3
	dstcr	0, r10
	dcp	[rp2], pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	shlb	row, 4, cr11
	stcr	1616928865, cr10
	addi32	cr11, col, cr11
	dstcr	0x0, pls.maskh, south
	cmplti32	cr11, 255, cr12
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x10, pls.count1, south
	dstcr	0x1, pls.count2, south
	stcr	0x2, bitwidthmode
	dstcr	0x0, plsthresholdsouth
	dstcr	0x100, pls.stride2, south
	dstcr	0x0, plsthresholdnorth
.LBB14_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB14_2 Depth 2
                                        //     Child Loop BB14_4 Depth 2
                                        //     Child Loop BB14_9 Depth 2
                                        //     Child Loop BB14_11 Depth 2
                                        //     Child Loop BB14_13 Depth 2
                                        //     Child Loop BB14_15 Depth 2
	dshlb	r10, 10, r11
	djmpincsetup	0, 16, :.LBB14_2
	daddi32	[rp3], r11, r11
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	cp	row, cr13
	cp	col, cr14
	dstcr	0x0, pc.constant, south
	dcp	r11, pls.addr, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB14_2:                               //   Parent Loop BB14_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB14_1 Depth=1
	djmpincsetup	0, 4, :.LBB14_4
	dstcr	0x200, pc.mode, south
.LBB14_4:                               //   Parent Loop BB14_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB14_1 Depth=1
	shlb	cr13, 4, cr15
	stcr	0, cr13
	addi32	cr15, cr14, cr14
	dstcr	0x200, pc.mode, south
	cmplti32	cr14, 255, cr14
	predpush	cr14, :.LBB14_7
// %bb.6:                               //   in Loop: Header=BB14_1 Depth=1
	cp	south, cr13
.LBB14_7:                               //   in Loop: Header=BB14_1 Depth=1
	predpop	
	dmuli32	r10, 255, r11
	dstcr	0x200, pc.mode, south
	dcpc	r11, cr14
	addi32	cr14, cr11, cr14
	predpush	cr12, :.LBB14_17
// %bb.8:                               //   in Loop: Header=BB14_1 Depth=1
	muli32hi	cr14, cr10, cr15
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	shrlb	cr15, 31, cr16
	shrab	cr15, 5, cr15
	dstcr	0x360, pc.mode, north
	addi32	cr15, cr16, cr15
	dcp	[rp2 + 1], dependencyid
	muli32	cr15, 11, cr15
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	addi32	cr15, cr14, cr14
	djmpincsetup	0, 4, :.LBB14_9
	shlb	cr14, 2, cr14
	dstcr	0x260, pc.mode, north
	nrb	cr14, north
.LBB14_9:                               //   Parent Loop BB14_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.10:                              //   in Loop: Header=BB14_1 Depth=1
	djmpincsetup	0, 16, :.LBB14_11
	dstcr	0x360, pc.mode, north
.LBB14_11:                              //   Parent Loop BB14_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.12:                              //   in Loop: Header=BB14_1 Depth=1
	djmpincsetup	0, 4, :.LBB14_13
	dstcr	0x260, pc.mode, north
	nrb	cr13, north
.LBB14_13:                              //   Parent Loop BB14_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB14_1 Depth=1
	djmpincsetup	0, 16, :.LBB14_15
	dstcr	0x360, pc.mode, north
.LBB14_15:                              //   Parent Loop BB14_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB14_1 Depth=1
	dstcr	0x260, pc.mode, north
.LBB14_17:                              // %Flow
                                        //   in Loop: Header=BB14_1 Depth=1
	predpop	
	djmpincne	r10, 169, :.LBB14_1
// %bb.18:
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z18fused_transpose_12I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj507ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj507EEEEvRT0_RT1_
_Z18fused_transpose_12I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj507ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj507EEEEvRT0_RT1_: // @_Z18fused_transpose_12I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj507ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj507EEEEvRT0_RT1_
// %bb.0:
	daddi32	rp1, -8, rp1
	addi32	crp1, -136, crp1        //     
	cp	row, cr10
	dcp	r11, rp2
	dcp	r10, rp3
	cp	col, cr11
	dstcr	0, r10
	addi32	crp1, 8, crp2           //      
	dstcr	268435424, r11
	stcr	507, cr12
	cmpeq32	cr10, 0, cr13
	dstcr	33686018, r12
	dstcr	2719, r13
	dstcr	0, r14
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	dstcr	0x2, mode
	stcr	0x2, bitwidthmode
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x200, pls.stride1, north
	dstcr	0x10, pls.stride2, north
.LBB15_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_2 Depth 2
                                        //       Child Loop BB15_4 Depth 3
                                        //       Child Loop BB15_6 Depth 3
                                        //       Child Loop BB15_8 Depth 3
                                        //       Child Loop BB15_10 Depth 3
                                        //       Child Loop BB15_12 Depth 3
                                        //     Child Loop BB15_16 Depth 2
                                        //       Child Loop BB15_17 Depth 3
	cp	crp2, crp3
	dstcr	0, r15
	dcp	[rp3], pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB15_2:                               //   Parent Loop BB15_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB15_4 Depth 3
                                        //       Child Loop BB15_6 Depth 3
                                        //       Child Loop BB15_8 Depth 3
                                        //       Child Loop BB15_10 Depth 3
                                        //       Child Loop BB15_12 Depth 3
	dshrab	r10, 31, r16
	stcr	0, cr14
	dshrlb	r16, 27, r16
	daddi32	r10, r16, r16
	dshrab	r16, 5, r17
	dandb	r16, r11, r16
	dcmplt32	r17, 85, r5
	dsubi32	r10, r16, r16
	dcpc	r17, cr15
	dshlb	r16, 4, r16
	dcpc	r5, cr16
	dcpc	r16, cr17
	addi32	cr17, cr11, cr17
	addi32	cr17, cr10, cr5
	cmplt32	cr17, cr12, cr17
	muli32	cr5, 96, cr5
	andb	cr13, cr17, cr17
	addi32	cr5, cr15, cr15
	andb	cr16, cr17, cr16
	predpush	cr16, :.LBB15_14
// %bb.3:                               //   in Loop: Header=BB15_2 Depth=2
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	[rp3 + 1], dependencyid
.LBB15_4:                               //   Parent Loop BB15_1 Depth=1
                                        //     Parent Loop BB15_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	dandb	r12, plsstatus, r16
	dorb	r16, pelsr, r16
	djmpneqoff	r16, 0, :.LBB15_4
// %bb.5:                               //   in Loop: Header=BB15_2 Depth=2
	shlb	cr15, 2, cr14
	djmpincsetup	0, 4, :.LBB15_6
	dstcr	0x1, plsstatus, north
	nrb	cr14, north
	dstcr	0x260, pc.mode, north
.LBB15_6:                               //   Parent Loop BB15_1 Depth=1
                                        //     Parent Loop BB15_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB15_2 Depth=2
	djmpincsetup	0, 16, :.LBB15_8
	dstcr	0x360, pc.mode, north
.LBB15_8:                               //   Parent Loop BB15_1 Depth=1
                                        //     Parent Loop BB15_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB15_2 Depth=2
	djmpincsetup	0, 16, :.LBB15_10
.LBB15_10:                              // %.preheader
                                        //   Parent Loop BB15_1 Depth=1
                                        //     Parent Loop BB15_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.11:                              //   in Loop: Header=BB15_2 Depth=2
	djmpincsetup	0, 4, :.LBB15_12
	dstcr	0x260, pc.mode, north
.LBB15_12:                              //   Parent Loop BB15_1 Depth=1
                                        //     Parent Loop BB15_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.13:                              //   in Loop: Header=BB15_2 Depth=2
	cp	north, cr14
.LBB15_14:                              // %Flow
                                        //   in Loop: Header=BB15_2 Depth=2
	predpop	
	daddi32	r10, 1, r10
	cp	cr14, [crp3+=1]
	djmpincne	r15, 32, :.LBB15_2
// %bb.15:                              //   in Loop: Header=BB15_1 Depth=1
	dshlb	r14, 6, r15
	cp	crp2, crp3
	daddi32	[rp2], r15, r17
	dstcr	0, r15
	dcp	[rp2 + 1], r16
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r17, pls.addr, north
	dstcr	0x1, pls.count1, north
	dstcr	0x20, pls.count2, north
	dcp	r16, dependencyid
	dstcr	0x1, plsstatus, north
.LBB15_16:                              //   Parent Loop BB15_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB15_17 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB15_17
	dstcr	0x200, pc.mode, north
.LBB15_17:                              //   Parent Loop BB15_1 Depth=1
                                        //     Parent Loop BB15_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB15_16 Depth=2
	addi32	crp3, 4, crp3
	dstcr	0x300, pc.mode, north
	nnb	south, north
	djmpincne	r15, 32, :.LBB15_16
// %bb.19:                              //   in Loop: Header=BB15_1 Depth=1
	daddi32	r14, 32, r14
	dstcr	0x200, pc.mode, north
	djmplte	r14, r13, :.LBB15_1
// %bb.20:
	daddi32	rp1, 8, rp1
	addi32	crp1, 136, crp1         //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z48fused_sigmoid_fixed_point_multiply_cast_cast_addI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_
_Z48fused_sigmoid_fixed_point_multiply_cast_cast_addI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_: // @_Z48fused_sigmoid_fixed_point_multiply_cast_cast_addI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -64, rp1
	daddi32	rp1, 56, rp2
	dstcr	0x2, mode
	addi32	crp1, -72, crp1         //     
	dcp	r11, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 48, rp2
	dcp	r10, rp4
	dcp	r9, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	507, r11
	dcp	r19, [rp2]
	dcp	r12, rp2
	dstcr	256, r12
	dstcr	522, r13
	addi32	crp1, 24, crp2          //      
	cp	crp1, crp3
	stcr	-65536, cr10
	stcr	65536, cr11
	stcr	16384, cr12
	dstcr	508, r14
	dcp	r20, [rp1 + 6]
	dcp	r21, [rp1 + 4]
	dcp	r22, [rp1 + 2]
	dcp	r23, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x200, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x200, pls.stride2, north
.LBB16_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_2 Depth 2
                                        //       Child Loop BB16_4 Depth 3
                                        //         Child Loop BB16_5 Depth 4
                                        //         Child Loop BB16_7 Depth 4
                                        //       Child Loop BB16_13 Depth 3
                                        //       Child Loop BB16_15 Depth 3
                                        //       Child Loop BB16_17 Depth 3
                                        //       Child Loop BB16_23 Depth 3
                                        //       Child Loop BB16_25 Depth 3
                                        //       Child Loop BB16_32 Depth 3
                                        //       Child Loop BB16_34 Depth 3
                                        //     Child Loop BB16_41 Depth 2
                                        //       Child Loop BB16_43 Depth 3
                                        //         Child Loop BB16_44 Depth 4
                                        //         Child Loop BB16_46 Depth 4
                                        //       Child Loop BB16_52 Depth 3
                                        //       Child Loop BB16_54 Depth 3
                                        //       Child Loop BB16_56 Depth 3
                                        //       Child Loop BB16_62 Depth 3
                                        //       Child Loop BB16_64 Depth 3
                                        //       Child Loop BB16_71 Depth 3
                                        //       Child Loop BB16_73 Depth 3
                                        //     Child Loop BB16_80 Depth 2
                                        //     Child Loop BB16_82 Depth 2
                                        //       Child Loop BB16_84 Depth 3
                                        //         Child Loop BB16_85 Depth 4
                                        //         Child Loop BB16_87 Depth 4
                                        //       Child Loop BB16_103 Depth 3
                                        //       Child Loop BB16_105 Depth 3
                                        //       Child Loop BB16_95 Depth 3
	dshlb	r10, 8, r15
	dcmpeq32	r10, 0, r16
	dsubi32	r11, r15, r17
	dcsel	r12, r11, r5
	dshrlb	r17, 8, r17
	dcp	[rp4], r6
	dandb	r17, 255, r29
	dstcr	0x11, pls.mode, south
	dcsel	1, r29, r29
	dstcr	0x300, pc.mode, south
	dshlb	r29, 8, r30
	dstcr	0x200, pc.mode, north
	daddi32	r30, r15, r8
	shlb	row, 4, cr13
	dsubi32	r5, r8, r30
	daddi32	r8, r12, r2
	daddi32	r30, 15, r9
	dandb	r30, 11, r19
	dshrab	r9, 31, r18
	dcmpeq32	r19, 0, r19
	dshrlb	r18, 28, r18
	dstcr	0, r7
	daddi32	r9, r18, r9
	dsubi32	r13, r8, r18
	dandb	r9, -16, r9
	dshrab	r18, 31, r19
	dcsel	r30, r9, r30
	dcmplt32	r5, r2, r2
	dshrab	r30, 31, r2
	dshrlb	r19, 28, r9
	dshrlb	r2, 28, r2
	daddi32	r18, r9, r9
	daddi32	r30, r2, r30
	dshrab	r9, 4, r9
	dshrab	r30, 4, r2
	dshrlb	r8, 4, r30
	dcsel	r2, 16, r2
	dcmpeq32	r8, 0, r8
	dcsel	1, r9, r9
	dcmpneq32	r16, 0, r16
	dstcr	0, r28
	addi32	cr13, col, cr13
	dsubi32	16, r2, r8
	dcsel	0, 251, r16
	stcr	0x2, bitwidthmode
	dstcr	0x0, pc.constant, south
.LBB16_2:                               //   Parent Loop BB16_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB16_4 Depth 3
                                        //         Child Loop BB16_5 Depth 4
                                        //         Child Loop BB16_7 Depth 4
                                        //       Child Loop BB16_13 Depth 3
                                        //       Child Loop BB16_15 Depth 3
                                        //       Child Loop BB16_17 Depth 3
                                        //       Child Loop BB16_23 Depth 3
                                        //       Child Loop BB16_25 Depth 3
                                        //       Child Loop BB16_32 Depth 3
                                        //       Child Loop BB16_34 Depth 3
	djmpeqoff	0, r29, :.LBB16_9
// %bb.3:                               //   in Loop: Header=BB16_2 Depth=2
	dshlb	r28, 2, r19
	dstcr	0, r18
	dcpc	r19, crp4
	addi32	crp2, crp4, crp4
.LBB16_4:                               // %.preheader17
                                        //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB16_5 Depth 4
                                        //         Child Loop BB16_7 Depth 4
	dshlb	r18, 8, r20
	dshlb	r7, 11, r19
	daddi32	r20, r15, r20
	daddi32	r6, r19, r19
	dshlb	r20, 2, r21
	dsubi32	r13, r20, r22
	daddi32	r19, r21, r19
	dshrab	r22, 31, r21
	dcmpeq32	r20, 0, r20
	dshrlb	r21, 28, r20
	dcp	r19, pls.addr, south
	daddi32	r22, r20, r20
	djmpincsetup	0, 16, :.LBB16_5
	dshrab	r20, 4, r19
	dcsel	16, r19, r19
	dcp	r19, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
.LBB16_5:                               //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        //       Parent Loop BB16_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB16_4 Depth=3
	djmpincsetup	0, 4, :.LBB16_7
	dstcr	0x200, pc.mode, south
.LBB16_7:                               //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        //       Parent Loop BB16_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB16_4 Depth=3
	cp	south, [crp4+=1]
	daddi32	r28, 1, r28
	djmpincne	r18, r29, :.LBB16_4
.LBB16_9:                               // %.loopexit18
                                        //   in Loop: Header=BB16_2 Depth=2
	djmpeqoff	0, r10, :.LBB16_39
// %bb.10:                              //   in Loop: Header=BB16_2 Depth=2
	dshlb	r7, 11, r18
	dshlb	r30, 6, r19
	daddi32	r6, r18, r20
	dstcr	1, r18
	daddi32	r20, r19, r19
                                        // implicit-def: $cx14
	dcp	r19, pls.addr, south
	dcp	r9, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r2, 0, :.LBB16_30
// %bb.11:                              //   in Loop: Header=BB16_2 Depth=2
	dstcr	1, r18
                                        // implicit-def: $cx14
	djmpeqoff	0, r8, :.LBB16_21
// %bb.12:                              //   in Loop: Header=BB16_2 Depth=2
	dcp	r2, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB16_13
.LBB16_13:                              // %.preheader16
                                        //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB16_2 Depth=2
	djmpincsetup	0, 4, :.LBB16_15
	dstcr	0x200, pc.mode, south
.LBB16_15:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB16_2 Depth=2
	dcp	r8, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB16_17
.LBB16_17:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB16_2 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB16_20
// %bb.19:                              //   in Loop: Header=BB16_2 Depth=2
	cp	south, cr14
.LBB16_20:                              // %Flow18
                                        //   in Loop: Header=BB16_2 Depth=2
	predpop	
	dstcr	0, r18
.LBB16_21:                              // %Flow20
                                        //   in Loop: Header=BB16_2 Depth=2
	djmpeqoff	0, r18, :.LBB16_29
// %bb.22:                              //   in Loop: Header=BB16_2 Depth=2
	dcp	r2, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB16_23
.LBB16_23:                              // %.preheader15
                                        //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB16_2 Depth=2
	djmpincsetup	0, 4, :.LBB16_25
	dstcr	0x200, pc.mode, south
.LBB16_25:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB16_2 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	dstcr	0x200, pc.mode, south
	predpush	cr15, :.LBB16_28
// %bb.27:                              //   in Loop: Header=BB16_2 Depth=2
	cp	south, cr14
.LBB16_28:                              // %Flow19
                                        //   in Loop: Header=BB16_2 Depth=2
	predpop	
.LBB16_29:                              // %Flow21
                                        //   in Loop: Header=BB16_2 Depth=2
	dstcr	0, r18
.LBB16_30:                              // %Flow23
                                        //   in Loop: Header=BB16_2 Depth=2
	djmpeqoff	r18, 0, :.LBB16_38
// %bb.31:                              //   in Loop: Header=BB16_2 Depth=2
	djmpincsetup	0, 4, :.LBB16_32
	dstcr	0x200, pc.mode, south
.LBB16_32:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB16_2 Depth=2
	dcp	r8, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB16_34
.LBB16_34:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.35:                              //   in Loop: Header=BB16_2 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB16_37
// %bb.36:                              //   in Loop: Header=BB16_2 Depth=2
	cp	south, cr14
.LBB16_37:                              // %Flow22
                                        //   in Loop: Header=BB16_2 Depth=2
	predpop	
.LBB16_38:                              //   in Loop: Header=BB16_2 Depth=2
	dshlb	r28, 2, r18
	daddi32	r28, 1, r28
	dcpc	r18, crp4
	addi32	crp2, crp4, crp4
	cp	cr14, [crp4]
.LBB16_39:                              //   in Loop: Header=BB16_2 Depth=2
	djmpincne	r7, 2, :.LBB16_2
// %bb.40:                              //   in Loop: Header=BB16_1 Depth=1
	dcmpeq32	r10, 0, r6
	dcsel	1, r17, r17
	dstcr	0x200, pc.mode, south
	dshlb	r17, 8, r6
	dstcr	0, r29
	daddi32	r6, r15, r6
	dstcr	0, r30
	dsubi32	r5, r6, r28
	daddi32	r6, r12, r7
	daddi32	r28, 15, r2
	dandb	r28, 11, r8
	dshrab	r2, 31, r9
	dcmpeq32	r8, 0, r8
	dshrlb	r9, 28, r8
	dsubi32	r13, r6, r9
	daddi32	r2, r8, r2
	dshrab	r9, 31, r8
	dandb	r2, -16, r2
	dshrlb	r8, 28, r8
	dcsel	r28, r2, r2
	dcmplt32	r5, r7, r5
	dshrab	r2, 31, r5
	daddi32	r9, r8, r28
	dshrlb	r5, 28, r5
	dcp	[rp3], r8
	daddi32	r2, r5, r5
	dstcr	0x9, pls.mode, south
	dshrab	r5, 4, r5
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	stcr	0x1, bitwidthmode
	dshrab	r28, 4, r28
	dcsel	r5, 16, r9
	shlb	row, 4, cr13
	dcmplti32	r6, 252, r5
	dshrlb	r6, 4, r2
	dsubi32	16, r9, r18
	dcsel	1, r28, r5
	addi32	cr13, col, cr13
	dstcr	0x0, pc.constant, south
.LBB16_41:                              //   Parent Loop BB16_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB16_43 Depth 3
                                        //         Child Loop BB16_44 Depth 4
                                        //         Child Loop BB16_46 Depth 4
                                        //       Child Loop BB16_52 Depth 3
                                        //       Child Loop BB16_54 Depth 3
                                        //       Child Loop BB16_56 Depth 3
                                        //       Child Loop BB16_62 Depth 3
                                        //       Child Loop BB16_64 Depth 3
                                        //       Child Loop BB16_71 Depth 3
                                        //       Child Loop BB16_73 Depth 3
	djmpeqoff	0, r17, :.LBB16_48
// %bb.42:                              //   in Loop: Header=BB16_41 Depth=2
	dshlb	r30, 1, r20
	dstcr	0, r19
	dcpc	r20, crp4
	addi32	crp3, crp4, crp4
.LBB16_43:                              // %.preheader13
                                        //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB16_44 Depth 4
                                        //         Child Loop BB16_46 Depth 4
	dshlb	r19, 8, r20
	dshlb	r29, 10, r21
	daddi32	r20, r15, r20
	daddi32	r8, r21, r21
	dshrab	r20, 31, r22
	dsubi32	r13, r20, r23
	dshrlb	r22, 28, r22
	djmpincsetup	0, 16, :.LBB16_44
	daddi32	r20, r22, r22
	dcmplti32	r20, 252, r20
	dshlb	r22, 1, r20
	dshrab	r23, 31, r22
	dandb	r20, -32, r20
	dshrlb	r22, 28, r22
	daddi32	r21, r20, r20
	daddi32	r23, r22, r21
	dcp	r20, pls.addr, south
	dshrab	r21, 4, r21
	dcsel	16, r21, r20
	dcp	r20, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB16_44:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        //       Parent Loop BB16_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.45:                              //   in Loop: Header=BB16_43 Depth=3
	djmpincsetup	0, 4, :.LBB16_46
	dstcr	0x200, pc.mode, south
.LBB16_46:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        //       Parent Loop BB16_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.47:                              //   in Loop: Header=BB16_43 Depth=3
	cp	south.0z, [crp4.z+=1]
	daddi32	r30, 1, r30
	djmpincne	r19, r17, :.LBB16_43
.LBB16_48:                              // %.loopexit14
                                        //   in Loop: Header=BB16_41 Depth=2
	djmpeqoff	0, r10, :.LBB16_78
// %bb.49:                              //   in Loop: Header=BB16_41 Depth=2
	dshlb	r29, 10, r19
	dshlb	r2, 5, r20
	daddi32	r8, r19, r21
	dstcr	1, r19
	daddi32	r21, r20, r20
                                        // implicit-def: $cx14
	dcp	r20, pls.addr, south
	dcp	r5, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r9, 0, :.LBB16_69
// %bb.50:                              //   in Loop: Header=BB16_41 Depth=2
	dstcr	1, r19
                                        // implicit-def: $cx14
	djmpeqoff	0, r18, :.LBB16_60
// %bb.51:                              //   in Loop: Header=BB16_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB16_52
.LBB16_52:                              // %.preheader12
                                        //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.53:                              //   in Loop: Header=BB16_41 Depth=2
	djmpincsetup	0, 4, :.LBB16_54
	dstcr	0x200, pc.mode, south
.LBB16_54:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.55:                              //   in Loop: Header=BB16_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB16_56
.LBB16_56:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB16_41 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB16_59
// %bb.58:                              //   in Loop: Header=BB16_41 Depth=2
	cp	south.0z, cr14
.LBB16_59:                              // %Flow9
                                        //   in Loop: Header=BB16_41 Depth=2
	predpop	
	dstcr	0, r19
.LBB16_60:                              // %Flow11
                                        //   in Loop: Header=BB16_41 Depth=2
	djmpeqoff	0, r19, :.LBB16_68
// %bb.61:                              //   in Loop: Header=BB16_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB16_62
.LBB16_62:                              // %.preheader11
                                        //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.63:                              //   in Loop: Header=BB16_41 Depth=2
	djmpincsetup	0, 4, :.LBB16_64
	dstcr	0x200, pc.mode, south
.LBB16_64:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB16_41 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	dstcr	0x200, pc.mode, south
	predpush	cr15, :.LBB16_67
// %bb.66:                              //   in Loop: Header=BB16_41 Depth=2
	cp	south.0z, cr14
.LBB16_67:                              // %Flow10
                                        //   in Loop: Header=BB16_41 Depth=2
	predpop	
.LBB16_68:                              // %Flow12
                                        //   in Loop: Header=BB16_41 Depth=2
	dstcr	0, r19
.LBB16_69:                              // %Flow14
                                        //   in Loop: Header=BB16_41 Depth=2
	djmpeqoff	r19, 0, :.LBB16_77
// %bb.70:                              //   in Loop: Header=BB16_41 Depth=2
	djmpincsetup	0, 4, :.LBB16_71
	dstcr	0x200, pc.mode, south
.LBB16_71:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.72:                              //   in Loop: Header=BB16_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB16_73
.LBB16_73:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.74:                              //   in Loop: Header=BB16_41 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB16_76
// %bb.75:                              //   in Loop: Header=BB16_41 Depth=2
	cp	south.0z, cr14
.LBB16_76:                              // %Flow13
                                        //   in Loop: Header=BB16_41 Depth=2
	predpop	
.LBB16_77:                              //   in Loop: Header=BB16_41 Depth=2
	dshlb	r30, 1, r19
	daddi32	r30, 1, r30
	dcpc	r19, crp4
	addi32	crp3, crp4, crp4
	cp	cr14, [crp4.z]
.LBB16_78:                              //   in Loop: Header=BB16_41 Depth=2
	djmpincne	r29, 2, :.LBB16_41
// %bb.79:                              //   in Loop: Header=BB16_1 Depth=1
	cp	crp3, crp4
	cp	crp2, crp5
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 2, :.LBB16_80
	dstcr	0x200, pc.mode, south
.LBB16_80:                              //   Parent Loop BB16_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp	[crp5], cr13
	muli32lohi{16}	cr10, cr13, cr13
	sfs	cr13, cr11
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr13
	addi32	cr13, cr11, cr13
	stcr	0x1, bitwidthmode
	divi32{16}	cr11, cr13, cr13
	shlb	[crp4.s+=1], 10, cr14
	muli32lohi{9}	cr13, cr12, cr13
	stcr	0x2, bitwidthmode
	addi32.lb	cr14, cr13, [crp5+=1]
// %bb.81:                              //   in Loop: Header=BB16_1 Depth=1
	dshrab	r6, 31, r29
	dcmplt32	r7, r14, r7
	dshrlb	r29, 28, r29
	dcsel	1, r28, r7
	dcp	[rp2], r28
	shlb	row, 4, cr13
	daddi32	r6, r29, r6
	addi32	cr13, col, cr13
	dshrab	r6, 4, r6
	dstcr	0, r29
	dstcr	0, r30
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
.LBB16_82:                              //   Parent Loop BB16_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB16_84 Depth 3
                                        //         Child Loop BB16_85 Depth 4
                                        //         Child Loop BB16_87 Depth 4
                                        //       Child Loop BB16_103 Depth 3
                                        //       Child Loop BB16_105 Depth 3
                                        //       Child Loop BB16_95 Depth 3
	djmpeqoff	r17, 0, :.LBB16_89
// %bb.83:                              //   in Loop: Header=BB16_82 Depth=2
	dshlb	r30, 2, r8
	addi32	crp1, 24, crp4          //      
	dstcr	0, r2
	dcpc	r8, crp5
	addi32	crp4, crp5, crp4
.LBB16_84:                              // %.preheader
                                        //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_82 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB16_85 Depth 4
                                        //         Child Loop BB16_87 Depth 4
	dshlb	r2, 8, r8
	dshlb	r29, 11, r9
	daddi32	r8, r15, r8
	daddi32	r28, r9, r9
	dshrab	r8, 31, r18
	dsubi32	r13, r8, r19
	dshrlb	r18, 28, r18
	djmpincsetup	0, 4, :.LBB16_85
	daddi32	r8, r18, r18
	dcmplti32	r8, 252, r8
	dshlb	r18, 2, r8
	dshrab	r19, 31, r18
	dandb	r8, -64, r8
	dshrlb	r18, 28, r18
	daddi32	r9, r8, r8
	daddi32	r19, r18, r9
	dcp	r8, pls.addr, north
	dshrab	r9, 4, r9
	dcsel	16, r9, r8
	dcp	r8, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	[crp4], north
	dstcr	0x200, pc.mode, north
.LBB16_85:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_82 Depth=2
                                        //       Parent Loop BB16_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.86:                              //   in Loop: Header=BB16_84 Depth=3
	djmpincsetup	0, 16, :.LBB16_87
	dstcr	0x300, pc.mode, north
.LBB16_87:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_82 Depth=2
                                        //       Parent Loop BB16_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.88:                              //   in Loop: Header=BB16_84 Depth=3
	addi32	crp4, 4, crp4
	daddi32	r30, 1, r30
	djmpincne	r2, r17, :.LBB16_84
.LBB16_89:                              // %.loopexit10
                                        //   in Loop: Header=BB16_82 Depth=2
	djmpeqoff	r10, 0, :.LBB16_97
// %bb.90:                              //   in Loop: Header=BB16_82 Depth=2
	dshlb	r29, 11, r2
	dshlb	r6, 6, r8
	daddi32	r28, r2, r2
	dshlb	r30, 2, r9
	daddi32	r2, r8, r8
	addi32	crp1, 24, crp4          //      
	dcpc	r9, crp5
	dcp	r8, pls.addr, north
	dcp	r5, pls.count1, north
	daddi32	r30, 1, r30
	dstcr	1, r2
	addi32	crp4, crp5, crp4
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r7, :.LBB16_91
// %bb.100:                             //   in Loop: Header=BB16_82 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr14
	predpush	cr14, :.LBB16_102
// %bb.101:                             //   in Loop: Header=BB16_82 Depth=2
	nrb	[crp4], north
.LBB16_102:                             //   in Loop: Header=BB16_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB16_103
	dstcr	0x200, pc.mode, north
.LBB16_103:                             //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.104:                             //   in Loop: Header=BB16_82 Depth=2
	dcp	r7, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB16_105
.LBB16_105:                             //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.106:                             // %Flow
                                        //   in Loop: Header=BB16_82 Depth=2
	dstcr	0, r2
.LBB16_91:                              // %Flow6
                                        //   in Loop: Header=BB16_82 Depth=2
	djmpeqoff	0, r2, :.LBB16_97
// %bb.92:                              //   in Loop: Header=BB16_82 Depth=2
	dcpc	r16, cr14
	cmplti32	cr13, cr14, cr14
	predpush	cr14, :.LBB16_94
// %bb.93:                              //   in Loop: Header=BB16_82 Depth=2
	nrb	[crp4], north
.LBB16_94:                              //   in Loop: Header=BB16_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB16_95
	dstcr	0x200, pc.mode, north
.LBB16_95:                              //   Parent Loop BB16_1 Depth=1
                                        //     Parent Loop BB16_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.96:                              //   in Loop: Header=BB16_82 Depth=2
	dstcr	0x300, pc.mode, north
.LBB16_97:                              // %.loopexit
                                        //   in Loop: Header=BB16_82 Depth=2
	djmpincne	r29, 2, :.LBB16_82
// %bb.98:                              //   in Loop: Header=BB16_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-421, r31
	djmpincne	r10, 2, r31
.LBB16_99:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r23
	dcp	[rp1 + 2], r22
	dcp	[rp1 + 4], r21
	dcp	[rp1 + 6], r20
	dcp	[rp2], r19
	daddi32	rp1, 40, rp2
	dcp	[rp2], r18
	daddi32	rp1, 48, rp2
	dcp	[rp2], r9
	daddi32	rp1, 56, rp2
	dcp	[rp2], r8
	daddi32	rp1, 64, rp1
	addi32	crp1, 72, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z23fused_exp_cast_multiplyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_
_Z23fused_exp_cast_multiplyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_: // @_Z23fused_exp_cast_multiplyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj507EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj507EEES6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -64, rp1
	daddi32	rp1, 56, rp2
	dstcr	0x2, mode
	addi32	crp1, -72, crp1         //     
	dcp	r11, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 48, rp2
	dcp	r10, rp4
	dcp	r9, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	507, r11
	dcp	r19, [rp2]
	dcp	r12, rp2
	dstcr	256, r12
	dstcr	522, r13
	addi32	crp1, 24, crp2          //      
	cp	crp1, crp3
	stcr	65536, cr10
	dstcr	508, r14
	dcp	r20, [rp1 + 6]
	dcp	r21, [rp1 + 4]
	dcp	r22, [rp1 + 2]
	dcp	r23, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x200, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x200, pls.stride2, north
.LBB17_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_2 Depth 2
                                        //       Child Loop BB17_4 Depth 3
                                        //         Child Loop BB17_5 Depth 4
                                        //         Child Loop BB17_7 Depth 4
                                        //       Child Loop BB17_13 Depth 3
                                        //       Child Loop BB17_15 Depth 3
                                        //       Child Loop BB17_17 Depth 3
                                        //       Child Loop BB17_23 Depth 3
                                        //       Child Loop BB17_25 Depth 3
                                        //       Child Loop BB17_32 Depth 3
                                        //       Child Loop BB17_34 Depth 3
                                        //     Child Loop BB17_41 Depth 2
                                        //       Child Loop BB17_43 Depth 3
                                        //         Child Loop BB17_44 Depth 4
                                        //         Child Loop BB17_46 Depth 4
                                        //       Child Loop BB17_52 Depth 3
                                        //       Child Loop BB17_54 Depth 3
                                        //       Child Loop BB17_56 Depth 3
                                        //       Child Loop BB17_62 Depth 3
                                        //       Child Loop BB17_64 Depth 3
                                        //       Child Loop BB17_71 Depth 3
                                        //       Child Loop BB17_73 Depth 3
                                        //     Child Loop BB17_80 Depth 2
                                        //     Child Loop BB17_82 Depth 2
                                        //       Child Loop BB17_84 Depth 3
                                        //         Child Loop BB17_85 Depth 4
                                        //         Child Loop BB17_87 Depth 4
                                        //       Child Loop BB17_103 Depth 3
                                        //       Child Loop BB17_105 Depth 3
                                        //       Child Loop BB17_95 Depth 3
	dshlb	r10, 8, r15
	dcmpeq32	r10, 0, r16
	dsubi32	r11, r15, r17
	dcsel	r12, r11, r5
	dshrlb	r17, 8, r17
	dcp	[rp4], r6
	dandb	r17, 255, r29
	dstcr	0x11, pls.mode, south
	dcsel	1, r29, r29
	dstcr	0x300, pc.mode, south
	dshlb	r29, 8, r30
	dstcr	0x200, pc.mode, north
	daddi32	r30, r15, r8
	shlb	row, 4, cr11
	dsubi32	r5, r8, r30
	daddi32	r8, r12, r2
	daddi32	r30, 15, r9
	dandb	r30, 11, r19
	dshrab	r9, 31, r18
	dcmpeq32	r19, 0, r19
	dshrlb	r18, 28, r18
	dstcr	0, r7
	daddi32	r9, r18, r9
	dsubi32	r13, r8, r18
	dandb	r9, -16, r9
	dshrab	r18, 31, r19
	dcsel	r30, r9, r30
	dcmplt32	r5, r2, r2
	dshrab	r30, 31, r2
	dshrlb	r19, 28, r9
	dshrlb	r2, 28, r2
	daddi32	r18, r9, r9
	daddi32	r30, r2, r30
	dshrab	r9, 4, r9
	dshrab	r30, 4, r2
	dshrlb	r8, 4, r30
	dcsel	r2, 16, r2
	dcmpeq32	r8, 0, r8
	dcsel	1, r9, r9
	dcmpneq32	r16, 0, r16
	dstcr	0, r28
	addi32	cr11, col, cr11
	dsubi32	16, r2, r8
	dcsel	0, 251, r16
	stcr	0x2, bitwidthmode
	dstcr	0x0, pc.constant, south
.LBB17_2:                               //   Parent Loop BB17_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB17_4 Depth 3
                                        //         Child Loop BB17_5 Depth 4
                                        //         Child Loop BB17_7 Depth 4
                                        //       Child Loop BB17_13 Depth 3
                                        //       Child Loop BB17_15 Depth 3
                                        //       Child Loop BB17_17 Depth 3
                                        //       Child Loop BB17_23 Depth 3
                                        //       Child Loop BB17_25 Depth 3
                                        //       Child Loop BB17_32 Depth 3
                                        //       Child Loop BB17_34 Depth 3
	djmpeqoff	0, r29, :.LBB17_9
// %bb.3:                               //   in Loop: Header=BB17_2 Depth=2
	dshlb	r28, 2, r19
	dstcr	0, r18
	dcpc	r19, crp4
	addi32	crp2, crp4, crp4
.LBB17_4:                               // %.preheader17
                                        //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB17_5 Depth 4
                                        //         Child Loop BB17_7 Depth 4
	dshlb	r18, 8, r20
	dshlb	r7, 11, r19
	daddi32	r20, r15, r20
	daddi32	r6, r19, r19
	dshlb	r20, 2, r21
	dsubi32	r13, r20, r22
	daddi32	r19, r21, r19
	dshrab	r22, 31, r21
	dcmpeq32	r20, 0, r20
	dshrlb	r21, 28, r20
	dcp	r19, pls.addr, south
	daddi32	r22, r20, r20
	djmpincsetup	0, 16, :.LBB17_5
	dshrab	r20, 4, r19
	dcsel	16, r19, r19
	dcp	r19, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
.LBB17_5:                               //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        //       Parent Loop BB17_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB17_4 Depth=3
	djmpincsetup	0, 4, :.LBB17_7
	dstcr	0x200, pc.mode, south
.LBB17_7:                               //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        //       Parent Loop BB17_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB17_4 Depth=3
	cp	south, [crp4+=1]
	daddi32	r28, 1, r28
	djmpincne	r18, r29, :.LBB17_4
.LBB17_9:                               // %.loopexit18
                                        //   in Loop: Header=BB17_2 Depth=2
	djmpeqoff	0, r10, :.LBB17_39
// %bb.10:                              //   in Loop: Header=BB17_2 Depth=2
	dshlb	r7, 11, r18
	dshlb	r30, 6, r19
	daddi32	r6, r18, r20
	dstcr	1, r18
	daddi32	r20, r19, r19
                                        // implicit-def: $cx12
	dcp	r19, pls.addr, south
	dcp	r9, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r2, 0, :.LBB17_30
// %bb.11:                              //   in Loop: Header=BB17_2 Depth=2
	dstcr	1, r18
                                        // implicit-def: $cx12
	djmpeqoff	0, r8, :.LBB17_21
// %bb.12:                              //   in Loop: Header=BB17_2 Depth=2
	dcp	r2, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB17_13
.LBB17_13:                              // %.preheader16
                                        //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB17_2 Depth=2
	djmpincsetup	0, 4, :.LBB17_15
	dstcr	0x200, pc.mode, south
.LBB17_15:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB17_2 Depth=2
	dcp	r8, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB17_17
.LBB17_17:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB17_2 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB17_20
// %bb.19:                              //   in Loop: Header=BB17_2 Depth=2
	cp	south, cr12
.LBB17_20:                              // %Flow18
                                        //   in Loop: Header=BB17_2 Depth=2
	predpop	
	dstcr	0, r18
.LBB17_21:                              // %Flow20
                                        //   in Loop: Header=BB17_2 Depth=2
	djmpeqoff	0, r18, :.LBB17_29
// %bb.22:                              //   in Loop: Header=BB17_2 Depth=2
	dcp	r2, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB17_23
.LBB17_23:                              // %.preheader15
                                        //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB17_2 Depth=2
	djmpincsetup	0, 4, :.LBB17_25
	dstcr	0x200, pc.mode, south
.LBB17_25:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB17_2 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	dstcr	0x200, pc.mode, south
	predpush	cr13, :.LBB17_28
// %bb.27:                              //   in Loop: Header=BB17_2 Depth=2
	cp	south, cr12
.LBB17_28:                              // %Flow19
                                        //   in Loop: Header=BB17_2 Depth=2
	predpop	
.LBB17_29:                              // %Flow21
                                        //   in Loop: Header=BB17_2 Depth=2
	dstcr	0, r18
.LBB17_30:                              // %Flow23
                                        //   in Loop: Header=BB17_2 Depth=2
	djmpeqoff	r18, 0, :.LBB17_38
// %bb.31:                              //   in Loop: Header=BB17_2 Depth=2
	djmpincsetup	0, 4, :.LBB17_32
	dstcr	0x200, pc.mode, south
.LBB17_32:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB17_2 Depth=2
	dcp	r8, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB17_34
.LBB17_34:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.35:                              //   in Loop: Header=BB17_2 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB17_37
// %bb.36:                              //   in Loop: Header=BB17_2 Depth=2
	cp	south, cr12
.LBB17_37:                              // %Flow22
                                        //   in Loop: Header=BB17_2 Depth=2
	predpop	
.LBB17_38:                              //   in Loop: Header=BB17_2 Depth=2
	dshlb	r28, 2, r18
	daddi32	r28, 1, r28
	dcpc	r18, crp4
	addi32	crp2, crp4, crp4
	cp	cr12, [crp4]
.LBB17_39:                              //   in Loop: Header=BB17_2 Depth=2
	djmpincne	r7, 2, :.LBB17_2
// %bb.40:                              //   in Loop: Header=BB17_1 Depth=1
	dcmpeq32	r10, 0, r6
	dcsel	1, r17, r17
	dstcr	0x200, pc.mode, south
	dshlb	r17, 8, r6
	dstcr	0, r29
	daddi32	r6, r15, r6
	dstcr	0, r30
	dsubi32	r5, r6, r28
	daddi32	r6, r12, r7
	daddi32	r28, 15, r2
	dandb	r28, 11, r8
	dshrab	r2, 31, r9
	dcmpeq32	r8, 0, r8
	dshrlb	r9, 28, r8
	dsubi32	r13, r6, r9
	daddi32	r2, r8, r2
	dshrab	r9, 31, r8
	dandb	r2, -16, r2
	dshrlb	r8, 28, r8
	dcsel	r28, r2, r2
	dcmplt32	r5, r7, r5
	dshrab	r2, 31, r5
	daddi32	r9, r8, r28
	dshrlb	r5, 28, r5
	dcp	[rp3], r8
	daddi32	r2, r5, r5
	dstcr	0x9, pls.mode, south
	dshrab	r5, 4, r5
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	stcr	0x1, bitwidthmode
	dshrab	r28, 4, r28
	dcsel	r5, 16, r9
	shlb	row, 4, cr11
	dcmplti32	r6, 252, r5
	dshrlb	r6, 4, r2
	dsubi32	16, r9, r18
	dcsel	1, r28, r5
	addi32	cr11, col, cr11
	dstcr	0x0, pc.constant, south
.LBB17_41:                              //   Parent Loop BB17_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB17_43 Depth 3
                                        //         Child Loop BB17_44 Depth 4
                                        //         Child Loop BB17_46 Depth 4
                                        //       Child Loop BB17_52 Depth 3
                                        //       Child Loop BB17_54 Depth 3
                                        //       Child Loop BB17_56 Depth 3
                                        //       Child Loop BB17_62 Depth 3
                                        //       Child Loop BB17_64 Depth 3
                                        //       Child Loop BB17_71 Depth 3
                                        //       Child Loop BB17_73 Depth 3
	djmpeqoff	0, r17, :.LBB17_48
// %bb.42:                              //   in Loop: Header=BB17_41 Depth=2
	dshlb	r30, 1, r20
	dstcr	0, r19
	dcpc	r20, crp4
	addi32	crp3, crp4, crp4
.LBB17_43:                              // %.preheader13
                                        //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB17_44 Depth 4
                                        //         Child Loop BB17_46 Depth 4
	dshlb	r19, 8, r20
	dshlb	r29, 10, r21
	daddi32	r20, r15, r20
	daddi32	r8, r21, r21
	dshrab	r20, 31, r22
	dsubi32	r13, r20, r23
	dshrlb	r22, 28, r22
	djmpincsetup	0, 16, :.LBB17_44
	daddi32	r20, r22, r22
	dcmplti32	r20, 252, r20
	dshlb	r22, 1, r20
	dshrab	r23, 31, r22
	dandb	r20, -32, r20
	dshrlb	r22, 28, r22
	daddi32	r21, r20, r20
	daddi32	r23, r22, r21
	dcp	r20, pls.addr, south
	dshrab	r21, 4, r21
	dcsel	16, r21, r20
	dcp	r20, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB17_44:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        //       Parent Loop BB17_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.45:                              //   in Loop: Header=BB17_43 Depth=3
	djmpincsetup	0, 4, :.LBB17_46
	dstcr	0x200, pc.mode, south
.LBB17_46:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        //       Parent Loop BB17_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.47:                              //   in Loop: Header=BB17_43 Depth=3
	cp	south.0z, [crp4.z+=1]
	daddi32	r30, 1, r30
	djmpincne	r19, r17, :.LBB17_43
.LBB17_48:                              // %.loopexit14
                                        //   in Loop: Header=BB17_41 Depth=2
	djmpeqoff	0, r10, :.LBB17_78
// %bb.49:                              //   in Loop: Header=BB17_41 Depth=2
	dshlb	r29, 10, r19
	dshlb	r2, 5, r20
	daddi32	r8, r19, r21
	dstcr	1, r19
	daddi32	r21, r20, r20
                                        // implicit-def: $cx12
	dcp	r20, pls.addr, south
	dcp	r5, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r9, 0, :.LBB17_69
// %bb.50:                              //   in Loop: Header=BB17_41 Depth=2
	dstcr	1, r19
                                        // implicit-def: $cx12
	djmpeqoff	0, r18, :.LBB17_60
// %bb.51:                              //   in Loop: Header=BB17_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB17_52
.LBB17_52:                              // %.preheader12
                                        //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.53:                              //   in Loop: Header=BB17_41 Depth=2
	djmpincsetup	0, 4, :.LBB17_54
	dstcr	0x200, pc.mode, south
.LBB17_54:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.55:                              //   in Loop: Header=BB17_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB17_56
.LBB17_56:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB17_41 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB17_59
// %bb.58:                              //   in Loop: Header=BB17_41 Depth=2
	cp	south.0z, cr12
.LBB17_59:                              // %Flow9
                                        //   in Loop: Header=BB17_41 Depth=2
	predpop	
	dstcr	0, r19
.LBB17_60:                              // %Flow11
                                        //   in Loop: Header=BB17_41 Depth=2
	djmpeqoff	0, r19, :.LBB17_68
// %bb.61:                              //   in Loop: Header=BB17_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB17_62
.LBB17_62:                              // %.preheader11
                                        //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.63:                              //   in Loop: Header=BB17_41 Depth=2
	djmpincsetup	0, 4, :.LBB17_64
	dstcr	0x200, pc.mode, south
.LBB17_64:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB17_41 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	dstcr	0x200, pc.mode, south
	predpush	cr13, :.LBB17_67
// %bb.66:                              //   in Loop: Header=BB17_41 Depth=2
	cp	south.0z, cr12
.LBB17_67:                              // %Flow10
                                        //   in Loop: Header=BB17_41 Depth=2
	predpop	
.LBB17_68:                              // %Flow12
                                        //   in Loop: Header=BB17_41 Depth=2
	dstcr	0, r19
.LBB17_69:                              // %Flow14
                                        //   in Loop: Header=BB17_41 Depth=2
	djmpeqoff	r19, 0, :.LBB17_77
// %bb.70:                              //   in Loop: Header=BB17_41 Depth=2
	djmpincsetup	0, 4, :.LBB17_71
	dstcr	0x200, pc.mode, south
.LBB17_71:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.72:                              //   in Loop: Header=BB17_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB17_73
.LBB17_73:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.74:                              //   in Loop: Header=BB17_41 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB17_76
// %bb.75:                              //   in Loop: Header=BB17_41 Depth=2
	cp	south.0z, cr12
.LBB17_76:                              // %Flow13
                                        //   in Loop: Header=BB17_41 Depth=2
	predpop	
.LBB17_77:                              //   in Loop: Header=BB17_41 Depth=2
	dshlb	r30, 1, r19
	daddi32	r30, 1, r30
	dcpc	r19, crp4
	addi32	crp3, crp4, crp4
	cp	cr12, [crp4.z]
.LBB17_78:                              //   in Loop: Header=BB17_41 Depth=2
	djmpincne	r29, 2, :.LBB17_41
// %bb.79:                              //   in Loop: Header=BB17_1 Depth=1
	cp	crp3, crp4
	cp	crp2, crp5
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 2, :.LBB17_80
	dstcr	0x200, pc.mode, south
.LBB17_80:                              //   Parent Loop BB17_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sfs	[crp5], cr10
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr11
	stcr	0x1, bitwidthmode
	shlb	[crp4.s+=1], 10, cr12
	stcr	0x2, bitwidthmode
	muli32lohi{16}.lb	cr11, cr12, [crp5+=1]
// %bb.81:                              //   in Loop: Header=BB17_1 Depth=1
	dshrab	r6, 31, r29
	dcmplt32	r7, r14, r7
	dshrlb	r29, 28, r29
	dcsel	1, r28, r7
	dcp	[rp2], r28
	shlb	row, 4, cr11
	daddi32	r6, r29, r6
	addi32	cr11, col, cr11
	dshrab	r6, 4, r6
	dstcr	0, r29
	dstcr	0, r30
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
.LBB17_82:                              //   Parent Loop BB17_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB17_84 Depth 3
                                        //         Child Loop BB17_85 Depth 4
                                        //         Child Loop BB17_87 Depth 4
                                        //       Child Loop BB17_103 Depth 3
                                        //       Child Loop BB17_105 Depth 3
                                        //       Child Loop BB17_95 Depth 3
	djmpeqoff	r17, 0, :.LBB17_89
// %bb.83:                              //   in Loop: Header=BB17_82 Depth=2
	dshlb	r30, 2, r8
	addi32	crp1, 24, crp4          //      
	dstcr	0, r2
	dcpc	r8, crp5
	addi32	crp4, crp5, crp4
.LBB17_84:                              // %.preheader
                                        //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_82 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB17_85 Depth 4
                                        //         Child Loop BB17_87 Depth 4
	dshlb	r2, 8, r8
	dshlb	r29, 11, r9
	daddi32	r8, r15, r8
	daddi32	r28, r9, r9
	dshrab	r8, 31, r18
	dsubi32	r13, r8, r19
	dshrlb	r18, 28, r18
	djmpincsetup	0, 4, :.LBB17_85
	daddi32	r8, r18, r18
	dcmplti32	r8, 252, r8
	dshlb	r18, 2, r8
	dshrab	r19, 31, r18
	dandb	r8, -64, r8
	dshrlb	r18, 28, r18
	daddi32	r9, r8, r8
	daddi32	r19, r18, r9
	dcp	r8, pls.addr, north
	dshrab	r9, 4, r9
	dcsel	16, r9, r8
	dcp	r8, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	[crp4], north
	dstcr	0x200, pc.mode, north
.LBB17_85:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_82 Depth=2
                                        //       Parent Loop BB17_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.86:                              //   in Loop: Header=BB17_84 Depth=3
	djmpincsetup	0, 16, :.LBB17_87
	dstcr	0x300, pc.mode, north
.LBB17_87:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_82 Depth=2
                                        //       Parent Loop BB17_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.88:                              //   in Loop: Header=BB17_84 Depth=3
	addi32	crp4, 4, crp4
	daddi32	r30, 1, r30
	djmpincne	r2, r17, :.LBB17_84
.LBB17_89:                              // %.loopexit10
                                        //   in Loop: Header=BB17_82 Depth=2
	djmpeqoff	r10, 0, :.LBB17_97
// %bb.90:                              //   in Loop: Header=BB17_82 Depth=2
	dshlb	r29, 11, r2
	dshlb	r6, 6, r8
	daddi32	r28, r2, r2
	dshlb	r30, 2, r9
	daddi32	r2, r8, r8
	addi32	crp1, 24, crp4          //      
	dcpc	r9, crp5
	dcp	r8, pls.addr, north
	dcp	r5, pls.count1, north
	daddi32	r30, 1, r30
	dstcr	1, r2
	addi32	crp4, crp5, crp4
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r7, :.LBB17_91
// %bb.100:                             //   in Loop: Header=BB17_82 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr12
	predpush	cr12, :.LBB17_102
// %bb.101:                             //   in Loop: Header=BB17_82 Depth=2
	nrb	[crp4], north
.LBB17_102:                             //   in Loop: Header=BB17_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB17_103
	dstcr	0x200, pc.mode, north
.LBB17_103:                             //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.104:                             //   in Loop: Header=BB17_82 Depth=2
	dcp	r7, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB17_105
.LBB17_105:                             //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.106:                             // %Flow
                                        //   in Loop: Header=BB17_82 Depth=2
	dstcr	0, r2
.LBB17_91:                              // %Flow6
                                        //   in Loop: Header=BB17_82 Depth=2
	djmpeqoff	0, r2, :.LBB17_97
// %bb.92:                              //   in Loop: Header=BB17_82 Depth=2
	dcpc	r16, cr12
	cmplti32	cr11, cr12, cr12
	predpush	cr12, :.LBB17_94
// %bb.93:                              //   in Loop: Header=BB17_82 Depth=2
	nrb	[crp4], north
.LBB17_94:                              //   in Loop: Header=BB17_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB17_95
	dstcr	0x200, pc.mode, north
.LBB17_95:                              //   Parent Loop BB17_1 Depth=1
                                        //     Parent Loop BB17_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.96:                              //   in Loop: Header=BB17_82 Depth=2
	dstcr	0x300, pc.mode, north
.LBB17_97:                              // %.loopexit
                                        //   in Loop: Header=BB17_82 Depth=2
	djmpincne	r29, 2, :.LBB17_82
// %bb.98:                              //   in Loop: Header=BB17_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-416, r31
	djmpincne	r10, 2, r31
.LBB17_99:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r23
	dcp	[rp1 + 2], r22
	dcp	[rp1 + 4], r21
	dcp	[rp1 + 6], r20
	dcp	[rp2], r19
	daddi32	rp1, 40, rp2
	dcp	[rp2], r18
	daddi32	rp1, 48, rp2
	dcp	[rp2], r9
	daddi32	rp1, 56, rp2
	dcp	[rp2], r8
	daddi32	rp1, 64, rp1
	addi32	crp1, 72, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z13fused_sigmoidI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj507EEES6_EvRT0_RT1_
_Z13fused_sigmoidI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj507EEES6_EvRT0_RT1_: // @_Z13fused_sigmoidI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj507EEES6_EvRT0_RT1_
// %bb.0:
	daddi32	rp1, -40, rp1
	addi32	crp1, -24, crp1         //     
	daddi32	rp1, 32, rp2
	dstcr	0x2, mode
	dcp	r10, rp3
	dstcr	0, r10
	dcp	r8, [rp2]
	dcp	r11, rp2
	dstcr	507, r11
	dstcr	256, r12
	dstcr	522, r13
	cp	crp1, crp2
	stcr	-65536, cr10
	stcr	65536, cr11
	dstcr	508, r14
	dcp	r9, [rp1 + 6]
	dcp	r18, [rp1 + 4]
	dcp	r19, [rp1 + 2]
	dcp	r20, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x200, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x200, pls.stride2, north
.LBB18_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_54 Depth 2
                                        //       Child Loop BB18_55 Depth 3
                                        //       Child Loop BB18_52 Depth 3
                                        //     Child Loop BB18_6 Depth 2
                                        //     Child Loop BB18_8 Depth 2
                                        //     Child Loop BB18_10 Depth 2
                                        //     Child Loop BB18_16 Depth 2
                                        //     Child Loop BB18_18 Depth 2
                                        //     Child Loop BB18_25 Depth 2
                                        //     Child Loop BB18_27 Depth 2
                                        //     Child Loop BB18_34 Depth 2
                                        //       Child Loop BB18_35 Depth 3
                                        //       Child Loop BB18_37 Depth 3
                                        //     Child Loop BB18_59 Depth 2
                                        //     Child Loop BB18_61 Depth 2
                                        //     Child Loop BB18_47 Depth 2
	dshlb	r10, 8, r16
	dcp	[rp3], r7
	dsubi32	r11, r16, r17
	dstcr	0x11, pls.mode, south
	dshrlb	r17, 8, r17
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dandb	r17, 255, r5
	shlb	row, 4, cr12
	dcmpeq32	r10, 0, r15
	dcsel	r12, r11, r6
	dcsel	0, 251, r15
	cp	crp2, crp3
	dstcr	0, r28
	dcsel	1, r5, r5
	addi32	cr12, col, cr12
	dstcr	0x0, pc.constant, south
	djmpeqoff	0, r5, :.LBB18_2
.LBB18_54:                              // %.preheader8
                                        //   Parent Loop BB18_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB18_55 Depth 3
                                        //       Child Loop BB18_52 Depth 3
	dshlb	r28, 8, r29
	djmpincsetup	0, 16, :.LBB18_55
	daddi32	r29, r16, r29
	dsubi32	r13, r29, r2
	dshlb	r29, 2, r30
	dcmpeq32	r29, 0, r29
	dshrab	r2, 31, r29
	daddi32	r7, r30, r30
	dshrlb	r29, 28, r29
	dcp	r30, pls.addr, south
	daddi32	r2, r29, r29
	dshrab	r29, 4, r29
	dcsel	16, r29, r29
	dcp	r29, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB18_55:                              //   Parent Loop BB18_1 Depth=1
                                        //     Parent Loop BB18_54 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.51:                              //   in Loop: Header=BB18_54 Depth=2
	djmpincsetup	0, 4, :.LBB18_52
	dstcr	0x200, pc.mode, south
.LBB18_52:                              //   Parent Loop BB18_1 Depth=1
                                        //     Parent Loop BB18_54 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.53:                              //   in Loop: Header=BB18_54 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r28, r5, :.LBB18_54
.LBB18_2:                               // %.loopexit9
                                        //   in Loop: Header=BB18_1 Depth=1
	djmpeqoff	0, r10, :.LBB18_32
// %bb.3:                               //   in Loop: Header=BB18_1 Depth=1
	dshlb	r5, 8, r29
	dstcr	1, r28
	daddi32	r29, r16, r29
                                        // implicit-def: $cx13
	dsubi32	r6, r29, r30
	dshrlb	r29, 4, r20
	daddi32	r30, 15, r8
	dsubi32	r13, r29, r19
	dshrab	r8, 31, r18
	daddi32	r29, r12, r2
	dshrlb	r18, 28, r18
	dcmpeq32	r29, 0, r29
	daddi32	r8, r18, r8
	dshlb	r20, 6, r18
	dandb	r30, 11, r9
	daddi32	r7, r18, r7
	dshrab	r19, 31, r18
	dcp	r7, pls.addr, south
	dshrlb	r18, 28, r29
	dandb	r8, -16, r8
	daddi32	r19, r29, r7
	dshrab	r7, 4, r7
	dcsel	1, r7, r7
	dcmpeq32	r9, 0, r29
	dcp	r7, pls.count1, south
	dcsel	r30, r8, r7
	dcmplt32	r6, r2, r6
	dshrab	r7, 31, r6
	dcp	[rp3 + 1], dependencyid
	dshrlb	r6, 28, r6
	dstcr	0x100, plsstatus, south
	daddi32	r7, r6, r6
	dcp	flowid, [rp3 + 1]
	dshrab	r6, 4, r6
	dcsel	r6, 16, r7
	dsubi32	16, r7, r6
	djmpeqoff	r7, 0, :.LBB18_23
// %bb.4:                               //   in Loop: Header=BB18_1 Depth=1
	dstcr	1, r28
                                        // implicit-def: $cx13
	dstcr	0x300, pc.mode, south
	djmpeqoff	0, r6, :.LBB18_14
// %bb.5:                               //   in Loop: Header=BB18_1 Depth=1
	dcp	r7, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB18_6
.LBB18_6:                               // %.preheader7
                                        //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB18_1 Depth=1
	djmpincsetup	0, 4, :.LBB18_8
	dstcr	0x200, pc.mode, south
.LBB18_8:                               //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB18_1 Depth=1
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB18_10
.LBB18_10:                              //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB18_1 Depth=1
	dcpc	r15, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	predpush	cr14, :.LBB18_13
// %bb.12:                              //   in Loop: Header=BB18_1 Depth=1
	cp	south, cr13
.LBB18_13:                              // %Flow8
                                        //   in Loop: Header=BB18_1 Depth=1
	predpop	
	dstcr	0, r28
.LBB18_14:                              // %Flow10
                                        //   in Loop: Header=BB18_1 Depth=1
	djmpeqoff	0, r28, :.LBB18_22
// %bb.15:                              //   in Loop: Header=BB18_1 Depth=1
	dcp	r7, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB18_16
.LBB18_16:                              // %.preheader6
                                        //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB18_1 Depth=1
	djmpincsetup	0, 4, :.LBB18_18
	dstcr	0x200, pc.mode, south
.LBB18_18:                              //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB18_1 Depth=1
	dcpc	r15, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	dstcr	0x200, pc.mode, south
	predpush	cr14, :.LBB18_21
// %bb.20:                              //   in Loop: Header=BB18_1 Depth=1
	cp	south, cr13
.LBB18_21:                              // %Flow9
                                        //   in Loop: Header=BB18_1 Depth=1
	predpop	
.LBB18_22:                              // %Flow11
                                        //   in Loop: Header=BB18_1 Depth=1
	dstcr	0, r28
.LBB18_23:                              // %Flow13
                                        //   in Loop: Header=BB18_1 Depth=1
	djmpeqoff	r28, 0, :.LBB18_31
// %bb.24:                              //   in Loop: Header=BB18_1 Depth=1
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, 4, :.LBB18_25
	dstcr	0x200, pc.mode, south
.LBB18_25:                              //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB18_1 Depth=1
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB18_27
.LBB18_27:                              //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB18_1 Depth=1
	dcpc	r15, cr13
	cmplti32	cr12, cr13, cr12
	stcr	0, cr13
	predpush	cr12, :.LBB18_30
// %bb.29:                              //   in Loop: Header=BB18_1 Depth=1
	cp	south, cr13
.LBB18_30:                              // %Flow12
                                        //   in Loop: Header=BB18_1 Depth=1
	predpop	
.LBB18_31:                              //   in Loop: Header=BB18_1 Depth=1
	dshlb	r5, 2, r5
	dcpc	r5, crp3
	addi32	crp2, crp3, crp3
	cp	cr13, [crp3]
.LBB18_32:                              //   in Loop: Header=BB18_1 Depth=1
	dstcr	0x200, pc.mode, south
	cp	[crp1], cr12
	dcmpeq32	r10, 0, r5
	muli32lohi{16}	cr10, cr12, cr12
	sfs	cr12, cr11
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr12
	addi32	cr12, cr11, cr13
	dcsel	1, r17, r17
	divi32{16}	cr11, cr13, [crp1]
	dcp	[rp2], r5
	shlb	row, 4, cr12
	addi32	cr12, col, cr12
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	djmpeqoff	r17, 0, :.LBB18_41
// %bb.33:                              //   in Loop: Header=BB18_1 Depth=1
	divi32{16}	cr11, cr13, cr13
	dstcr	0, r6
	orb	crp2, 4, crp3
.LBB18_34:                              // %.preheader
                                        //   Parent Loop BB18_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB18_35 Depth 3
                                        //       Child Loop BB18_37 Depth 3
	dshlb	r6, 8, r7
	djmpincsetup	0, 4, :.LBB18_35
	daddi32	r7, r16, r7
	dshrab	r7, 31, r28
	dsubi32	r13, r7, r29
	dshrlb	r28, 28, r28
	dshrab	r29, 31, r30
	daddi32	r7, r28, r28
	dshrlb	r30, 28, r30
	dshlb	r28, 2, r28
	daddi32	r29, r30, r29
	dandb	r28, -64, r28
	dshrab	r29, 4, r29
	dcmplti32	r7, 252, r7
	daddi32	r5, r28, r28
	dcsel	16, r29, r7
	dcp	r28, pls.addr, north
	dcp	r7, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	cr13, north
	dstcr	0x200, pc.mode, north
.LBB18_35:                              //   Parent Loop BB18_1 Depth=1
                                        //     Parent Loop BB18_34 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.36:                              //   in Loop: Header=BB18_34 Depth=2
	djmpincsetup	0, 16, :.LBB18_37
	dstcr	0x300, pc.mode, north
.LBB18_37:                              //   Parent Loop BB18_1 Depth=1
                                        //     Parent Loop BB18_34 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.38:                              //   in Loop: Header=BB18_34 Depth=2
	daddi32	r6, 1, r6
	dstcr	1, r7
                                        // implicit-def: $cx13
	djmpeqoff	r6, r17, :.LBB18_40
// %bb.39:                              //   in Loop: Header=BB18_34 Depth=2
	cp	[crp3+=1], cr13
	dstcr	0, r7
.LBB18_40:                              // %Flow6
                                        //   in Loop: Header=BB18_34 Depth=2
	djmpeqoff	r7, 0, :.LBB18_34
.LBB18_41:                              // %.loopexit5
                                        //   in Loop: Header=BB18_1 Depth=1
	djmpeqoff	r10, 0, :.LBB18_49
// %bb.42:                              //   in Loop: Header=BB18_1 Depth=1
	dshlb	r17, 8, r6
	dshlb	r17, 2, r7
	daddi32	r6, r16, r16
	dstcr	1, r17
	dshrab	r16, 31, r6
	dcpc	r7, crp3
	dshrlb	r6, 28, r6
	dsubi32	r13, r16, r7
	daddi32	r16, r6, r6
	daddi32	r16, r12, r28
	dcmplti32	r16, 252, r16
	dshrab	r6, 4, r16
	dshrab	r7, 31, r6
	dshlb	r16, 6, r16
	dshrlb	r6, 28, r6
	daddi32	r5, r16, r5
	daddi32	r7, r6, r16
	dcp	r5, pls.addr, north
	dshrab	r16, 4, r16
	addi32	crp2, crp3, crp3
	dcsel	1, r16, r6
	dcmplt32	r28, r14, r7
	dcp	r6, pls.count1, north
	dcsel	1, r16, r16
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r16, :.LBB18_43
// %bb.56:                              //   in Loop: Header=BB18_1 Depth=1
	dcpc	r15, cr13
	cmplti32	cr12, cr13, cr13
	predpush	cr13, :.LBB18_58
// %bb.57:                              //   in Loop: Header=BB18_1 Depth=1
	nrb	[crp3], north
.LBB18_58:                              //   in Loop: Header=BB18_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB18_59
	dstcr	0x200, pc.mode, north
.LBB18_59:                              //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.60:                              //   in Loop: Header=BB18_1 Depth=1
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB18_61
.LBB18_61:                              //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.62:                              // %Flow
                                        //   in Loop: Header=BB18_1 Depth=1
	dstcr	0, r17
.LBB18_43:                              // %Flow4
                                        //   in Loop: Header=BB18_1 Depth=1
	djmpeqoff	0, r17, :.LBB18_49
// %bb.44:                              //   in Loop: Header=BB18_1 Depth=1
	dcpc	r15, cr13
	cmplti32	cr12, cr13, cr12
	predpush	cr12, :.LBB18_46
// %bb.45:                              //   in Loop: Header=BB18_1 Depth=1
	nrb	[crp3], north
.LBB18_46:                              //   in Loop: Header=BB18_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB18_47
	dstcr	0x200, pc.mode, north
.LBB18_47:                              //   Parent Loop BB18_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.48:                              //   in Loop: Header=BB18_1 Depth=1
	dstcr	0x300, pc.mode, north
.LBB18_49:                              // %.loopexit
                                        //   in Loop: Header=BB18_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-261, r31
	djmpincne	r10, 2, r31
.LBB18_50:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r20
	dcp	[rp1 + 2], r19
	dcp	[rp1 + 4], r18
	dcp	[rp1 + 6], r9
	dcp	[rp2], r8
	daddi32	rp1, 40, rp1
	addi32	crp1, 24, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z15fused_sigmoid_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj507EEES6_EvRT0_RT1_
_Z15fused_sigmoid_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj507EEES6_EvRT0_RT1_: // @_Z15fused_sigmoid_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj507EEES6_EvRT0_RT1_
// %bb.0:
	daddi32	rp1, -64, rp1
	daddi32	rp1, 56, rp2
	dstcr	0x2, mode
	stcr	-336, cr10
	dcp	r10, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 48, rp2
	addi32	crp1, cr10, crp1        //     
	dcp	r9, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	256, r12
	dcp	r19, [rp2]
	dcp	r11, rp2
	dstcr	507, r11
	dstcr	522, r13
	addi32	crp1, 16, crp2          //      
	stcr	-65536, cr10
	stcr	65536, cr11
	dstcr	508, r14
	dcp	r20, [rp1 + 6]
	dcp	r21, [rp1 + 4]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x200, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x200, pls.stride2, north
.LBB19_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_2 Depth 2
                                        //       Child Loop BB19_4 Depth 3
                                        //         Child Loop BB19_5 Depth 4
                                        //         Child Loop BB19_7 Depth 4
                                        //       Child Loop BB19_13 Depth 3
                                        //       Child Loop BB19_15 Depth 3
                                        //       Child Loop BB19_17 Depth 3
                                        //       Child Loop BB19_23 Depth 3
                                        //       Child Loop BB19_25 Depth 3
                                        //       Child Loop BB19_32 Depth 3
                                        //       Child Loop BB19_34 Depth 3
                                        //     Child Loop BB19_41 Depth 2
                                        //     Child Loop BB19_43 Depth 2
                                        //       Child Loop BB19_45 Depth 3
                                        //         Child Loop BB19_46 Depth 4
                                        //         Child Loop BB19_48 Depth 4
                                        //       Child Loop BB19_64 Depth 3
                                        //       Child Loop BB19_66 Depth 3
                                        //       Child Loop BB19_56 Depth 3
	dshlb	r10, 8, r15
	dcmpeq32	r10, 0, r16
	dsubi32	r11, r15, r17
	dcsel	r12, r11, r29
	dshrlb	r17, 8, r17
	dcp	[rp3], r5
	dandb	r17, 255, r28
	dstcr	0x11, pls.mode, south
	dcsel	1, r28, r28
	dstcr	0x300, pc.mode, south
	dshlb	r28, 8, r30
	dstcr	0x200, pc.mode, north
	daddi32	r30, r15, r2
	shlb	row, 4, cr12
	dsubi32	r29, r2, r30
	daddi32	r2, r12, r8
	daddi32	r30, 15, r9
	dandb	r30, 11, r18
	dshrab	r9, 31, r19
	dcmpeq32	r18, 0, r18
	dshrlb	r19, 28, r18
	dsubi32	r13, r2, r19
	daddi32	r9, r18, r9
	dshrab	r19, 31, r18
	dandb	r9, -16, r9
	dshrlb	r18, 28, r18
	dcsel	r30, r9, r30
	dcmplt32	r29, r8, r29
	dshrab	r30, 31, r29
	daddi32	r19, r18, r8
	dshrlb	r29, 28, r29
	dshrab	r8, 4, r8
	daddi32	r30, r29, r30
	dshrlb	r2, 4, r29
	dshrab	r30, 4, r30
	dstcr	0, r6
	dcsel	r30, 16, r30
	dcmpeq32	r2, 0, r2
	dcsel	1, r8, r8
	dcmpneq32	r16, 0, r16
	dstcr	0, r7
	addi32	cr12, col, cr12
	dsubi32	16, r30, r2
	dcsel	0, 251, r16
	dstcr	0x0, pc.constant, south
.LBB19_2:                               //   Parent Loop BB19_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB19_4 Depth 3
                                        //         Child Loop BB19_5 Depth 4
                                        //         Child Loop BB19_7 Depth 4
                                        //       Child Loop BB19_13 Depth 3
                                        //       Child Loop BB19_15 Depth 3
                                        //       Child Loop BB19_17 Depth 3
                                        //       Child Loop BB19_23 Depth 3
                                        //       Child Loop BB19_25 Depth 3
                                        //       Child Loop BB19_32 Depth 3
                                        //       Child Loop BB19_34 Depth 3
	djmpeqoff	0, r28, :.LBB19_9
// %bb.3:                               //   in Loop: Header=BB19_2 Depth=2
	dshlb	r7, 2, r18
	dstcr	0, r9
	dcpc	r18, crp3
	addi32	crp2, crp3, crp3
.LBB19_4:                               // %.preheader10
                                        //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB19_5 Depth 4
                                        //         Child Loop BB19_7 Depth 4
	dshlb	r9, 8, r19
	dshlb	r6, 11, r18
	daddi32	r19, r15, r19
	daddi32	r5, r18, r18
	dshlb	r19, 2, r20
	dsubi32	r13, r19, r21
	daddi32	r18, r20, r18
	dshrab	r21, 31, r20
	dcmpeq32	r19, 0, r19
	dshrlb	r20, 28, r19
	dcp	r18, pls.addr, south
	daddi32	r21, r19, r19
	djmpincsetup	0, 16, :.LBB19_5
	dshrab	r19, 4, r18
	dcsel	16, r18, r18
	dcp	r18, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB19_5:                               //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        //       Parent Loop BB19_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB19_4 Depth=3
	djmpincsetup	0, 4, :.LBB19_7
	dstcr	0x200, pc.mode, south
.LBB19_7:                               //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        //       Parent Loop BB19_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB19_4 Depth=3
	cp	south, [crp3+=1]
	daddi32	r7, 1, r7
	djmpincne	r9, r28, :.LBB19_4
.LBB19_9:                               // %.loopexit11
                                        //   in Loop: Header=BB19_2 Depth=2
	djmpeqoff	0, r10, :.LBB19_39
// %bb.10:                              //   in Loop: Header=BB19_2 Depth=2
	dshlb	r6, 11, r9
	dshlb	r29, 6, r18
	daddi32	r5, r9, r19
	dstcr	1, r9
	daddi32	r19, r18, r18
                                        // implicit-def: $cx13
	dcp	r18, pls.addr, south
	dcp	r8, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r30, 0, :.LBB19_30
// %bb.11:                              //   in Loop: Header=BB19_2 Depth=2
	dstcr	1, r9
                                        // implicit-def: $cx13
	djmpeqoff	0, r2, :.LBB19_21
// %bb.12:                              //   in Loop: Header=BB19_2 Depth=2
	dcp	r30, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB19_13
.LBB19_13:                              // %.preheader9
                                        //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB19_2 Depth=2
	djmpincsetup	0, 4, :.LBB19_15
	dstcr	0x200, pc.mode, south
.LBB19_15:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB19_2 Depth=2
	dcp	r2, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB19_17
.LBB19_17:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB19_2 Depth=2
	dcpc	r16, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	predpush	cr14, :.LBB19_20
// %bb.19:                              //   in Loop: Header=BB19_2 Depth=2
	cp	south, cr13
.LBB19_20:                              // %Flow7
                                        //   in Loop: Header=BB19_2 Depth=2
	predpop	
	dstcr	0, r9
.LBB19_21:                              // %Flow9
                                        //   in Loop: Header=BB19_2 Depth=2
	djmpeqoff	0, r9, :.LBB19_29
// %bb.22:                              //   in Loop: Header=BB19_2 Depth=2
	dcp	r30, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB19_23
.LBB19_23:                              // %.preheader8
                                        //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB19_2 Depth=2
	djmpincsetup	0, 4, :.LBB19_25
	dstcr	0x200, pc.mode, south
.LBB19_25:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB19_2 Depth=2
	dcpc	r16, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	dstcr	0x200, pc.mode, south
	predpush	cr14, :.LBB19_28
// %bb.27:                              //   in Loop: Header=BB19_2 Depth=2
	cp	south, cr13
.LBB19_28:                              // %Flow8
                                        //   in Loop: Header=BB19_2 Depth=2
	predpop	
.LBB19_29:                              // %Flow10
                                        //   in Loop: Header=BB19_2 Depth=2
	dstcr	0, r9
.LBB19_30:                              // %Flow12
                                        //   in Loop: Header=BB19_2 Depth=2
	djmpeqoff	r9, 0, :.LBB19_38
// %bb.31:                              //   in Loop: Header=BB19_2 Depth=2
	djmpincsetup	0, 4, :.LBB19_32
	dstcr	0x200, pc.mode, south
.LBB19_32:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB19_2 Depth=2
	dcp	r2, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB19_34
.LBB19_34:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.35:                              //   in Loop: Header=BB19_2 Depth=2
	dcpc	r16, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	predpush	cr14, :.LBB19_37
// %bb.36:                              //   in Loop: Header=BB19_2 Depth=2
	cp	south, cr13
.LBB19_37:                              // %Flow11
                                        //   in Loop: Header=BB19_2 Depth=2
	predpop	
.LBB19_38:                              //   in Loop: Header=BB19_2 Depth=2
	dshlb	r7, 2, r9
	daddi32	r7, 1, r7
	dcpc	r9, crp3
	addi32	crp2, crp3, crp3
	cp	cr13, [crp3]
.LBB19_39:                              //   in Loop: Header=BB19_2 Depth=2
	djmpincne	r6, 80, :.LBB19_2
// %bb.40:                              //   in Loop: Header=BB19_1 Depth=1
	cp	crp2, crp3
	djmpincsetup	0, 80, :.LBB19_41
	dstcr	0x200, pc.mode, south
.LBB19_41:                              //   Parent Loop BB19_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp	[crp3], cr12
	muli32lohi{16}	cr10, cr12, cr12
	sfs	cr12, cr11
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr12
	addi32	cr12, cr11, cr12
	divi32{16}.lb	cr11, cr12, [crp3+=1]
// %bb.42:                              //   in Loop: Header=BB19_1 Depth=1
	dcmpeq32	r10, 0, r5
	dcsel	1, r17, r17
	dcp	[rp2], r5
	dshlb	r17, 8, r6
	shlb	row, 4, cr12
	daddi32	r6, r15, r7
	addi32	cr12, col, cr12
	dsubi32	r13, r7, r6
	daddi32	r7, r12, r28
	dshrab	r6, 31, r29
	dcmplt32	r28, r14, r28
	dshrlb	r29, 28, r28
	dshrab	r7, 31, r29
	daddi32	r6, r28, r6
	dshrlb	r29, 28, r28
	dshrab	r6, 4, r29
	daddi32	r7, r28, r28
	dcsel	1, r29, r6
	dcmplti32	r7, 252, r7
	dcsel	1, r29, r7
	dshrab	r28, 4, r28
	dstcr	0, r29
	dstcr	0, r30
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
.LBB19_43:                              //   Parent Loop BB19_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB19_45 Depth 3
                                        //         Child Loop BB19_46 Depth 4
                                        //         Child Loop BB19_48 Depth 4
                                        //       Child Loop BB19_64 Depth 3
                                        //       Child Loop BB19_66 Depth 3
                                        //       Child Loop BB19_56 Depth 3
	djmpeqoff	r17, 0, :.LBB19_50
// %bb.44:                              //   in Loop: Header=BB19_43 Depth=2
	dshlb	r30, 2, r8
	dstcr	0, r2
	dcpc	r8, crp3
	addi32	crp2, crp3, crp3
.LBB19_45:                              // %.preheader
                                        //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_43 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB19_46 Depth 4
                                        //         Child Loop BB19_48 Depth 4
	dshlb	r2, 8, r8
	dshlb	r29, 11, r9
	daddi32	r8, r15, r8
	daddi32	r5, r9, r9
	dshrab	r8, 31, r18
	dsubi32	r13, r8, r19
	dshrlb	r18, 28, r18
	djmpincsetup	0, 4, :.LBB19_46
	daddi32	r8, r18, r18
	dcmplti32	r8, 252, r8
	dshlb	r18, 2, r8
	dshrab	r19, 31, r18
	dandb	r8, -64, r8
	dshrlb	r18, 28, r18
	daddi32	r9, r8, r8
	daddi32	r19, r18, r9
	dcp	r8, pls.addr, north
	dshrab	r9, 4, r9
	dcsel	16, r9, r8
	dcp	r8, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	[crp3], north
	dstcr	0x200, pc.mode, north
.LBB19_46:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_43 Depth=2
                                        //       Parent Loop BB19_45 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.47:                              //   in Loop: Header=BB19_45 Depth=3
	djmpincsetup	0, 16, :.LBB19_48
	dstcr	0x300, pc.mode, north
.LBB19_48:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_43 Depth=2
                                        //       Parent Loop BB19_45 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.49:                              //   in Loop: Header=BB19_45 Depth=3
	addi32	crp3, 4, crp3
	daddi32	r30, 1, r30
	djmpincne	r2, r17, :.LBB19_45
.LBB19_50:                              // %.loopexit7
                                        //   in Loop: Header=BB19_43 Depth=2
	djmpeqoff	r10, 0, :.LBB19_58
// %bb.51:                              //   in Loop: Header=BB19_43 Depth=2
	dshlb	r29, 11, r2
	dshlb	r28, 6, r8
	daddi32	r5, r2, r2
	dshlb	r30, 2, r9
	daddi32	r2, r8, r8
	addi32	crp1, 16, crp3          //      
	dcpc	r9, crp4
	dcp	r8, pls.addr, north
	dcp	r7, pls.count1, north
	daddi32	r30, 1, r30
	dstcr	1, r2
	addi32	crp3, crp4, crp3
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r6, :.LBB19_52
// %bb.61:                              //   in Loop: Header=BB19_43 Depth=2
	dcpc	r16, cr13
	cmplti32	cr12, cr13, cr13
	predpush	cr13, :.LBB19_63
// %bb.62:                              //   in Loop: Header=BB19_43 Depth=2
	nrb	[crp3], north
.LBB19_63:                              //   in Loop: Header=BB19_43 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB19_64
	dstcr	0x200, pc.mode, north
.LBB19_64:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_43 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB19_43 Depth=2
	dcp	r6, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB19_66
.LBB19_66:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_43 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.67:                              // %Flow
                                        //   in Loop: Header=BB19_43 Depth=2
	dstcr	0, r2
.LBB19_52:                              // %Flow4
                                        //   in Loop: Header=BB19_43 Depth=2
	djmpeqoff	0, r2, :.LBB19_58
// %bb.53:                              //   in Loop: Header=BB19_43 Depth=2
	dcpc	r16, cr13
	cmplti32	cr12, cr13, cr13
	predpush	cr13, :.LBB19_55
// %bb.54:                              //   in Loop: Header=BB19_43 Depth=2
	nrb	[crp3], north
.LBB19_55:                              //   in Loop: Header=BB19_43 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB19_56
	dstcr	0x200, pc.mode, north
.LBB19_56:                              //   Parent Loop BB19_1 Depth=1
                                        //     Parent Loop BB19_43 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB19_43 Depth=2
	dstcr	0x300, pc.mode, north
.LBB19_58:                              // %.loopexit
                                        //   in Loop: Header=BB19_43 Depth=2
	djmpincne	r29, 80, :.LBB19_43
// %bb.59:                              //   in Loop: Header=BB19_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-281, r31
	djmpincne	r10, 2, r31
.LBB19_60:
	daddi32	rp1, 32, rp2
	dcp	[rp1 + 4], r21
	dcp	[rp1 + 6], r20
	dcp	[rp2], r19
	daddi32	rp1, 40, rp2
	dcp	[rp2], r18
	daddi32	rp1, 48, rp2
	dcp	[rp2], r9
	daddi32	rp1, 56, rp2
	dcp	[rp2], r8
	daddi32	rp1, 64, rp1
	stcr	336, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z102fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_16215914359837010491_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj34816EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj128ELj13ELj13EEEEvRT0_RT1_RT2_
_Z102fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_16215914359837010491_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj34816EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj128ELj13ELj13EEEEvRT0_RT1_RT2_: // @_Z102fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_16215914359837010491_I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj13ELj13EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj34816EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj128ELj13ELj13EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-528, cr10
	dcp	r10, rp4
	addi32	crp1, cr10, crp1        //     
	dstcr	0x2, mode
	stcr	272, crp2
	dcp	r11, rp3
	dcp	[rp4], r11
	addi32	crp1, crp2, crp2
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dstcr	0x0, pls.maskh, south
	dstcr	0x1fff0, pls.maskl, south
	dcp	r11, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0xd, pls.count1, south
	dstcr	0x40, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xd0, pls.stride2, south
	dcp	r12, rp2
	dstcr	0, r10
	cp	crp2, crp3
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	stcr	0x2, bitwidthmode
.LBB20_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_2 Depth 2
                                        //     Child Loop BB20_4 Depth 2
                                        //     Child Loop BB20_6 Depth 2
	djmpincsetup	0, 4, :.LBB20_2
	dstcr	0x200, pc.mode, south
.LBB20_2:                               //   Parent Loop BB20_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB20_1 Depth=1
	djmpincsetup	0, 13, :.LBB20_4
	dstcr	0x300, pc.mode, south
.LBB20_4:                               //   Parent Loop BB20_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB20_1 Depth=1
	djmpincsetup	0, 7, :.LBB20_6
	dstcr	0x200, pc.mode, south
.LBB20_6:                               //   Parent Loop BB20_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB20_1 Depth=1
	cp	south, [crp3+=1]
	djmpincne	r10, 64, :.LBB20_1
// %bb.8:
	dstcr	0x200, pc.mode, south
	stcr	0x0, vapmode
	dcp	[rp3], r11
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dcp	r11, pls.addr, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x1100, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0, r10
	addi32	crp1, 16, crp3          //      
	stcr	14975, cr10
	stcr	11755, cr11
	stcr	12527, cr12
	stcr	6553, cr13
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB20_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_10 Depth 2
	cp	crp2, crp4
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 32, :.LBB20_10
	stcr	0x0, accumall
.LBB20_10:                              //   Parent Loop BB20_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	macwrxi8.lb	[crp4+=2]
// %bb.11:                              //   in Loop: Header=BB20_9 Depth=1
	accsumsh8	0
	cp	accum0, cr14
	cp	accum0h, cr15
	addi32	cr14, cr15, cr14
	muli32lohi{22}	cr14, cr10, cr14
	mini32	cr14, 127, cr14
	maxi32	cr14, -127, cr14
	muli32	>wl, cr14, cr14
	addi32	>wl, cr14, cr14
	stcr	0x1, bitwidthmode
	muli32lohi{20}	cr14, cr11, cr14
	mini32	cr14, 127, cr14
	maxi32	cr14, -127, cr14
	shlb	cr14, 16, cr14
	muli32lohi{17}	cr14, cr12, cr14
	muli32lohi{16}	cr14, cr13, cr15
	cmplti32	0, cr14, cr16
	csel	cr14, cr15, cr14
	shrlb	cr14, 10, [crp3.z+=1]
	djmpincne	r10, 128, :.LBB20_9
// %bb.12:
	dcp	[rp2], r11
	dstcr	0x8, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dcp	r11, pls.addr, north
	dstcr	0x20, pls.stride1, north
	dstcr	0xd, pls.count1, north
	dstcr	0x80, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x1a0, pls.stride2, north
	addi32	crp1, 16, crp2          //      
	dstcr	0, r10
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
.LBB20_13:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_14 Depth 2
                                        //     Child Loop BB20_16 Depth 2
	nrb	[crp2.z], north
	djmpincsetup	0, 4, :.LBB20_14
	dstcr	0x200, pc.mode, north
.LBB20_14:                              //   Parent Loop BB20_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.15:                              //   in Loop: Header=BB20_13 Depth=1
	djmpincsetup	0, 13, :.LBB20_16
	dstcr	0x300, pc.mode, north
.LBB20_16:                              //   Parent Loop BB20_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB20_13 Depth=1
	addi32	crp2, 2, crp2
	djmpincne	r10, 128, :.LBB20_13
// %bb.18:
	dstcr	0x200, pc.mode, north
	daddi32	rp1, 16, rp1
	stcr	528, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z30fused_nn_conv2d_transpose_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS1_L10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IS1_LS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj1536EEES2_IS0_IiLh16ELh4ELi0EELS3_0EjLj64ELS4_1EJLj1ELj128ELj26ELj26EEEEvRT0_RT1_RT2_
_Z30fused_nn_conv2d_transpose_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS1_L10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IS1_LS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj1536EEES2_IS0_IiLh16ELh4ELi0EELS3_0EjLj64ELS4_1EJLj1ELj128ELj26ELj26EEEEvRT0_RT1_RT2_: // @_Z30fused_nn_conv2d_transpose_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS1_L10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj128ELj13ELj13EEES2_IS1_LS3_0EjLj64ELS4_1EJLj1ELj1ELj1ELj1536EEES2_IS0_IiLh16ELh4ELi0EELS3_0EjLj64ELS4_1EJLj1ELj128ELj26ELj26EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-832, cr10
	stcr	528, crp2
	addi32	crp1, cr10, crp1        //     
	dcp	r12, rp2
	dcp	r11, rp3
	dcp	r10, rp4
	dstcr	0, r10
	addi32	crp1, crp2, crp2
	dstcr	0x0, pls.maskh, south
	dstcr	0x1fff0, pls.maskl, south
	dstcr	0x20, pls.stride1, south
	dstcr	0x80, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x1a0, pls.stride2, south
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x180, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x20, pls.stride1, north
	dstcr	0x2, mode
	dstcr	0x80, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x340, pls.stride2, north
.LBB21_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_5 Depth 2
                                        //       Child Loop BB21_6 Depth 3
                                        //       Child Loop BB21_8 Depth 3
                                        //       Child Loop BB21_10 Depth 3
                                        //     Child Loop BB21_15 Depth 2
                                        //       Child Loop BB21_16 Depth 3
                                        //       Child Loop BB21_18 Depth 3
                                        //     Child Loop BB21_24 Depth 2
                                        //       Child Loop BB21_25 Depth 3
                                        //       Child Loop BB21_27 Depth 3
                                        //     Child Loop BB21_32 Depth 2
                                        //       Child Loop BB21_33 Depth 3
                                        //     Child Loop BB21_40 Depth 2
                                        //       Child Loop BB21_41 Depth 3
                                        //       Child Loop BB21_43 Depth 3
                                        //     Child Loop BB21_48 Depth 2
                                        //       Child Loop BB21_49 Depth 3
                                        //     Child Loop BB21_55 Depth 2
                                        //       Child Loop BB21_56 Depth 3
                                        //     Child Loop BB21_61 Depth 2
                                        //     Child Loop BB21_64 Depth 2
                                        //       Child Loop BB21_65 Depth 3
                                        //     Child Loop BB21_68 Depth 2
                                        //       Child Loop BB21_72 Depth 3
                                        //       Child Loop BB21_78 Depth 3
                                        //       Child Loop BB21_82 Depth 3
                                        //       Child Loop BB21_84 Depth 3
                                        //     Child Loop BB21_87 Depth 2
                                        //     Child Loop BB21_89 Depth 2
                                        //       Child Loop BB21_90 Depth 3
                                        //       Child Loop BB21_92 Depth 3
	dshrlb	r10, 1, r11
	dstcr	1, r16
	dshlb	r11, 3, r12
	dcmpeq32	r11, 0, r14
	dcsel	r12, 4, r13
	dcmpneq32	r11, 0, r15
	dsubi32	r12, r13, r17
	dsubi32	5, r12, r5
	dcsel	r17, 0, r17
	dmaxi32	r5, 0, r5
	dshlb	r17, 6, r17
	dsubi32	13, r12, r6
	daddi32	[rp4], r17, r17
	dstcr	0x69, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	daddi32	[rp4 + 3], r5, r5
	dshlb	r15, 2, r15
	dmuli32	r5, r14, r14
	dsubi32	4, r13, r13
	dmin32	r6, 8, r6
	dxorb	r15, 20, r15
	dcsel	r13, 4, r13
	daddi32	r14, r6, r14
	dsubi32	4, r13, r7
	dmin32	r14, r15, r14
	dcp	r17, pls.addr, south
	daddi32	r14, r7, r15
	dsubi32	20, r14, r14
	dcp	r15, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r13, 0, :.LBB21_36
// %bb.2:                               //   in Loop: Header=BB21_1 Depth=1
	dstcr	1, r16
	djmpeqoff	r15, 0, :.LBB21_21
// %bb.3:                               //   in Loop: Header=BB21_1 Depth=1
	dstcr	1, r17
	dstcr	0, r16
	cp	crp2, crp3
	djmpeqoff	0, r14, :.LBB21_13
// %bb.4:                               //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
.LBB21_5:                               // %.preheader27
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_6 Depth 3
                                        //       Child Loop BB21_8 Depth 3
                                        //       Child Loop BB21_10 Depth 3
	dcp	r13, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB21_6
.LBB21_6:                               //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB21_5 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB21_8
.LBB21_8:                               //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB21_5 Depth=2
	dcp	r14, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB21_10
.LBB21_10:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB21_5 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r16, 128, :.LBB21_5
// %bb.12:                              // %Flow7
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r17
.LBB21_13:                              // %Flow9
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r16
	cp	crp2, crp3
	djmpeqoff	r17, 0, :.LBB21_20
// %bb.14:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
.LBB21_15:                              // %.preheader25
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_16 Depth 3
                                        //       Child Loop BB21_18 Depth 3
	dcp	r13, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB21_16
.LBB21_16:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB21_15 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB21_18
.LBB21_18:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB21_15 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r16, 128, :.LBB21_15
.LBB21_20:                              // %Flow10
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r16
.LBB21_21:                              // %Flow15
                                        //   in Loop: Header=BB21_1 Depth=1
	djmpeqoff	r16, 0, :.LBB21_35
// %bb.22:                              //   in Loop: Header=BB21_1 Depth=1
	dstcr	1, r17
	dstcr	0, r16
	cp	crp2, crp3
	djmpeqoff	0, r14, :.LBB21_30
// %bb.23:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB21_24:                              // %.preheader23
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_25 Depth 3
                                        //       Child Loop BB21_27 Depth 3
	dcp	r13, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB21_25
.LBB21_25:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB21_24 Depth=2
	dcp	r14, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB21_27
.LBB21_27:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB21_24 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r16, 128, :.LBB21_24
// %bb.29:                              // %Flow11
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r17
.LBB21_30:                              // %Flow13
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r16
	cp	crp2, crp3
	djmpeqoff	r17, 0, :.LBB21_35
// %bb.31:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB21_32:                              // %.preheader21
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_33 Depth 3
	dcp	r13, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB21_33
.LBB21_33:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB21_32 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r16, 128, :.LBB21_32
.LBB21_35:                              // %Flow16
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r16
.LBB21_36:                              // %Flow27
                                        //   in Loop: Header=BB21_1 Depth=1
	dandb	r10, 1, r13
	djmpeqoff	r16, 0, :.LBB21_62
// %bb.37:                              //   in Loop: Header=BB21_1 Depth=1
	dstcr	1, r16
	djmpeqoff	r15, 0, :.LBB21_52
// %bb.38:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	528, crp3
	dstcr	1, r17
	addi32	crp1, crp3, crp3
	dstcr	0, r16
	cp	crp3, crp4
	djmpeqoff	0, r14, :.LBB21_46
// %bb.39:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
.LBB21_40:                              // %.preheader19
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_41 Depth 3
                                        //       Child Loop BB21_43 Depth 3
	dcp	r15, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB21_41
.LBB21_41:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB21_40 Depth=2
	dcp	r14, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB21_43
.LBB21_43:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB21_40 Depth=2
	cp	south.0z, [crp4.z+=1]
	djmpincne	r16, 128, :.LBB21_40
// %bb.45:                              // %Flow17
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r17
.LBB21_46:                              // %Flow19
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r16
	djmpeqoff	r17, 0, :.LBB21_51
// %bb.47:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB21_48:                              // %.preheader17
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_49 Depth 3
	dcp	r15, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB21_49
.LBB21_49:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB21_48 Depth=2
	cp	south.0z, [crp3.z+=1]
	djmpincne	r16, 128, :.LBB21_48
.LBB21_51:                              // %Flow20
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r16
.LBB21_52:                              // %Flow25
                                        //   in Loop: Header=BB21_1 Depth=1
	djmpeqoff	r16, 0, :.LBB21_62
// %bb.53:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	528, crp3
	dstcr	1, r16
	addi32	crp1, crp3, crp3
	dstcr	0, r15
	cp	crp3, crp4
	djmpeqoff	0, r14, :.LBB21_59
// %bb.54:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB21_55:                              // %.preheader15
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_56 Depth 3
	dcp	r14, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB21_56
.LBB21_56:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB21_55 Depth=2
	cp	south.0z, [crp4.z+=1]
	djmpincne	r15, 128, :.LBB21_55
// %bb.58:                              // %Flow21
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0, r16
.LBB21_59:                              // %Flow23
                                        //   in Loop: Header=BB21_1 Depth=1
	djmpeqoff	r16, 0, :.LBB21_62
// %bb.60:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	0x1, bitwidthmode
	djmpincsetup	0, 128, :.LBB21_61
.LBB21_61:                              // %.preheader13
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south.0z, [crp3.z+=1]
.LBB21_62:                              // %.loopexit14
                                        //   in Loop: Header=BB21_1 Depth=1
	dstcr	0x200, pc.mode, south
	addi32	col, 1, cr10
	dshlb	r13, 3, r14
	shrlb	cr10, 31, cr11
	addi32	cr10, cr11, cr10
	shrab	cr10, 1, cr10
	subi32	col, cr10, cr10
	addi32	row, 1, cr11
	shrlb	cr11, 31, cr12
	addi32	cr11, cr12, cr11
	shrab	cr11, 1, cr11
	subi32	row, cr11, cr11
	djmpeqoff	r13, 0, :.LBB21_67
// %bb.63:                              //   in Loop: Header=BB21_1 Depth=1
	stcr	528, crp3
	dstcr	0, r15
	addi32	crp1, crp3, crp3
	stcr	0x1, bitwidthmode
.LBB21_64:                              // %.preheader
                                        //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_65 Depth 3
	dcp	r14, jumpendcount
	cp	[crp3.z], cr12
	djmpincsetup	0, jumpendcount, :.LBB21_65
.LBB21_65:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_64 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nrb	cr12, west
	cp.lb	east.0z, cr12
// %bb.66:                              //   in Loop: Header=BB21_64 Depth=2
	cp	cr12, [crp3.z+=1]
	djmpincne	r15, 128, :.LBB21_64
.LBB21_67:                              // %.loopexit
                                        //   in Loop: Header=BB21_1 Depth=1
	stcr	784, crp4
	shlb	cr10, 1, crp3
	addi32	crp1, crp4, crp4
	shlb	cr11, 1, crp5
	addi32	crp4, crp3, crp3
	stcr	808, crp4
	dstcr	0, r15
	addi32	crp1, crp4, crp4
	stcr	0x1, bitwidthmode
	addi32	crp4, crp5, crp4
	stcr	528, crp5
	addi32	crp1, crp5, crp5
.LBB21_68:                              //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_72 Depth 3
                                        //       Child Loop BB21_78 Depth 3
                                        //       Child Loop BB21_82 Depth 3
                                        //       Child Loop BB21_84 Depth 3
	stcr	808, crp6
	dshrlb	r15, 3, r16
	cp	[crp5.z], cr12
	addi32	crp1, crp6, crp6
	dandb	r16, 16, r16
	cp	cr12, [crp6.z]
	stcr	0, [crp5.z]
	cp	row, cr10
	dorb	r16, r14, r16
	dcpc	r12, cr13
	cp	col, cr11
	nrb	cr12, east
	cp	west.0z, cr14
	addi32	cr10, cr13, cr13
	dcpc	r16, cr15
	nrb	cr14, south
	cmpeq32	cr13, 13, cr13
	cmplti32	cr10, 9, cr14
	addi32	cr11, cr15, cr15
	andb	cr14, cr13, cr13
	cmplti32	8, cr11, cr14
	cmpneq32	cr15, 13, cr15
	orb	cr14, cr15, cr14
	predpush	cr14, :.LBB21_69
// %bb.96:                              //   in Loop: Header=BB21_68 Depth=2
	nrb	cr12, south
	predpush	cr13, :.LBB21_98
// %bb.97:                              //   in Loop: Header=BB21_68 Depth=2
	cp	north.0z, cr12
.LBB21_98:                              // %Flow4
                                        //   in Loop: Header=BB21_68 Depth=2
	predpop	
.LBB21_69:                              // %Flow5
                                        //   in Loop: Header=BB21_68 Depth=2
	predelse	:.LBB21_71
// %bb.70:                              //   in Loop: Header=BB21_68 Depth=2
	cmpneq32	cr13, 0, cr13
	cp	west.0z, cr12
	cp	north.0z, cr14
	csel	cr14, cr12, cr12
	nrb	cr12, south
.LBB21_71:                              //   in Loop: Header=BB21_68 Depth=2
	predpop	
	stcr	808, crp6
	djmpincsetup	1, 9, :.LBB21_72
	addi32	crp1, crp6, crp6
	cp	cr12, [crp6.z]
	stcr	808, crp6
	nrb	cr12, south
	addi32	crp1, crp6, crp6
	orb	crp6, 2, crp7
.LBB21_72:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_68 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	cp	north.0z, [crp7.z+=1]
	cp	north.0z, cr12
	nrb.lb	cr12, south
// %bb.73:                              //   in Loop: Header=BB21_68 Depth=2
	andb	cr10, 1, cr13
                                        // implicit-def: $cx12
	predpush	cr13, :.LBB21_75
// %bb.74:                              //   in Loop: Header=BB21_68 Depth=2
	stcr	784, crp7
	addi32	crp1, crp7, crp7
	cp	[crp7.z], cr12
.LBB21_75:                              // %Flow
                                        //   in Loop: Header=BB21_68 Depth=2
	predelse	:.LBB21_77
// %bb.76:                              //   in Loop: Header=BB21_68 Depth=2
	stcr	784, crp7
	cp	[crp4.z], cr12
	addi32	crp1, crp7, crp7
	cp	cr12, [crp7.z]
.LBB21_77:                              //   in Loop: Header=BB21_68 Depth=2
	predpop	
	stcr	784, crp7
	nrb	cr12, east
	addi32	crp1, crp7, crp7
	djmpincsetup	1, 9, :.LBB21_78
	orb	crp7, 2, crp7
.LBB21_78:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_68 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	cp	west.0z, [crp7.z+=1]
	cp	west.0z, cr12
	nrb.lb	cr12, east
// %bb.79:                              //   in Loop: Header=BB21_68 Depth=2
	orb	cr11, cr10, cr11
	stcr	0, cr10
	andb	cr11, 1, cr11
	cmpeq32	cr11, 0, cr11
	predpush	cr11, :.LBB21_81
// %bb.80:                              //   in Loop: Header=BB21_68 Depth=2
	cp	[crp3.z], cr10
	cp	cr10, [crp5.z]
.LBB21_81:                              // %.preheader58
                                        //   in Loop: Header=BB21_68 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB21_82
.LBB21_82:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_68 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nrb	cr10, north
	cp.lb	south.0z, cr10
// %bb.83:                              //   in Loop: Header=BB21_68 Depth=2
	cp	cr10, [crp5.z]
	djmpincsetup	0, 4, :.LBB21_84
.LBB21_84:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_68 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nrb	cr10, south
	cp.lb	north.0z, cr10
// %bb.85:                              //   in Loop: Header=BB21_68 Depth=2
	cp	cr10, [crp5.z+=1]
	djmpincne	r15, 128, :.LBB21_68
// %bb.86:                              //   in Loop: Header=BB21_1 Depth=1
	dcp	[rp3], r12
	stcr	528, crp3
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r12, pls.addr, west
	addi32	crp1, crp3, crp3
	addi32	crp1, 16, crp4          //      
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
	djmpincsetup	0, 128, :.LBB21_87
.LBB21_87:                              //   Parent Loop BB21_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	stcr	808, crp5
	stcr	0x1, bitwidthmode
	addi32	crp1, crp5, crp5
	cp	[crp3+=1], cr10
	cp	cr10, [crp5.z]
	cp	crp6, crp5
	stcr	0x0, accumall
	stcr	0x2, bitwidthmode
	macwrxi	[crp5+=2]
	stcr	0x1, bitwidthmode
	addi32	crp5, -8, crp5
	nrb	[crp5.z+=1], north | south | east | west
	macwrni		<>	nnbr	r90
	macwrni	
	accsumsh	6
	cp	accum0, cr10
	shlb	cr10, 16, cr10
	shrab	cr10, 16, cr10
	addi32	cr10, col, cr10
	subi32	cr10, col, cr10
	stcr	0x2, bitwidthmode
	shlb.lb	cr10, 10, [crp4+=1]
// %bb.88:                              //   in Loop: Header=BB21_1 Depth=1
	dshlb	r11, 11, r12
	dshlb	r13, 6, r13
	daddi32	[rp2], r12, r12
	dcmpeq32	r11, 0, r11
	daddi32	r12, r13, r13
	dcsel	16, 10, r11
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r13, pls.addr, north
	dcp	r11, pls.count1, north
	dstcr	0, r12
	addi32	crp1, 16, crp3          //      
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
.LBB21_89:                              //   Parent Loop BB21_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB21_90 Depth 3
                                        //       Child Loop BB21_92 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB21_90
	dstcr	0x200, pc.mode, north
.LBB21_90:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_89 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.91:                              //   in Loop: Header=BB21_89 Depth=2
	dcp	r11, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB21_92
.LBB21_92:                              //   Parent Loop BB21_1 Depth=1
                                        //     Parent Loop BB21_89 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.93:                              //   in Loop: Header=BB21_89 Depth=2
	addi32	crp3, 4, crp3
	djmpincne	r12, 128, :.LBB21_89
// %bb.94:                              //   in Loop: Header=BB21_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-329, r31
	djmpincne	r10, 4, r31
.LBB21_95:
	daddi32	rp1, 16, rp1
	stcr	832, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z47fused_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj384ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj96ELj26ELj26EEEEvRT0_RT1_
_Z47fused_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj384ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj96ELj26ELj26EEEEvRT0_RT1_: // @_Z47fused_fixed_point_multiply_cast_round_clip_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj384ELj26ELj26EEES2_IDv4_aLS4_0EjLj64ELS5_1EJLj1ELj96ELj26ELj26EEEEvRT0_RT1_
// %bb.0:
	daddi32	rp1, -24, rp1
	stcr	-1936, cr10
	stcr	400, crp2
	addi32	crp1, cr10, crp1        //     
	dcp	r11, rp2
	dcp	r10, rp3
	dstcr	0, r10
	dstcr	-1, r11
	addi32	crp1, crp2, crp2
	dstcr	384, r12
	stcr	15282, cr10
	stcr	-32768, cr11
	stcr	32768, cr12
	stcr	8323072, cr13
	stcr	-8323072, cr14
	dstcr	0x2, mode
	dcp	r8, [rp1 + 4]
	dstcr	0x20, pls.stride1, south
	dstcr	0x180, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x340, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x20, pls.stride1, north
	dstcr	0x60, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x340, pls.stride2, north
.LBB22_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_5 Depth 2
                                        //       Child Loop BB22_6 Depth 3
                                        //       Child Loop BB22_8 Depth 3
                                        //       Child Loop BB22_10 Depth 3
                                        //     Child Loop BB22_15 Depth 2
                                        //       Child Loop BB22_16 Depth 3
                                        //       Child Loop BB22_18 Depth 3
                                        //     Child Loop BB22_24 Depth 2
                                        //       Child Loop BB22_25 Depth 3
                                        //       Child Loop BB22_27 Depth 3
                                        //     Child Loop BB22_32 Depth 2
                                        //       Child Loop BB22_33 Depth 3
                                        //     Child Loop BB22_40 Depth 2
                                        //       Child Loop BB22_41 Depth 3
                                        //       Child Loop BB22_43 Depth 3
                                        //     Child Loop BB22_48 Depth 2
                                        //       Child Loop BB22_49 Depth 3
                                        //     Child Loop BB22_55 Depth 2
                                        //       Child Loop BB22_56 Depth 3
                                        //     Child Loop BB22_61 Depth 2
                                        //     Child Loop BB22_63 Depth 2
                                        //     Child Loop BB22_65 Depth 2
                                        //       Child Loop BB22_66 Depth 3
                                        //       Child Loop BB22_68 Depth 3
	dshrlb	r10, 1, r14
	dshlb	r10, 4, r15
	dcmpeq32	r14, 0, r13
	dshlb	r14, 4, r13
	dandb	r15, 16, r15
	dcsel	r13, 4, r16
	dcmpneq32	r14, 0, r17
	dsubi32	r13, r16, r14
	dsubi32	10, r13, r5
	dcsel	r14, 0, r14
	dmaxi32	r5, 0, r7
	dsubi32	26, r15, r5
	dshlb	r14, 7, r14
	dmini32	r5, 20, r5
	daddi32	[rp3], r14, r14
	dshlb	r15, 2, r6
	dsubi32	60, r5, r28
	daddi32	r14, r6, r29
	dshrlb	r11, r28, r14
	dcmplti32	28, r5, r6
	dcsel	r14, 0, r28
	dsubi32	28, r5, r14
	dcmplti32	r5, 28, r5
	dshrlb	r11, r14, r5
	daddi32	r13, 16, r14
	dcsel	r5, -1, r5
	dcmpeq32	r15, 0, r30
	dandb	r5, -16, r6
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcsel	r6, r5, r30
	dcmplt32	r14, 26, r2
	daddi32	[rp3 + 3], r7, r7
	dsubi32	26, r13, r5
	dcmplt32	25, r14, r6
	dmuli32	r7, r2, r7
	dsubi32	4, r16, r16
	dmin32	r5, 16, r8
	dshlb	r6, 2, r5
	dcmpneq32	r17, 0, r17
	dxorb	r5, 20, r17
	dcsel	r16, 4, r5
	daddi32	r7, r8, r7
	dsubi32	4, r5, r16
	dmin32	r7, r17, r7
	dcp	r28, pls.maskh, south
	daddi32	r7, r16, r17
	dcp	r30, pls.maskl, south
	dcp	r29, pls.addr, south
	dcp	r17, pls.count1, south
	dstcr	1, r6
	dsubi32	20, r7, r16
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	djmpeqoff	r5, 0, :.LBB22_36
// %bb.2:                               //   in Loop: Header=BB22_1 Depth=1
	dstcr	1, r6
	djmpeqoff	r17, 0, :.LBB22_21
// %bb.3:                               //   in Loop: Header=BB22_1 Depth=1
	dstcr	1, r7
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	0, r16, :.LBB22_13
// %bb.4:                               //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB22_5:                               // %.preheader46
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_6 Depth 3
                                        //       Child Loop BB22_8 Depth 3
                                        //       Child Loop BB22_10 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB22_6
.LBB22_6:                               //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB22_5 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB22_8
.LBB22_8:                               //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB22_5 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB22_10
.LBB22_10:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB22_5 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, r12, :.LBB22_5
// %bb.12:                              // %Flow
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r7
.LBB22_13:                              // %Flow4
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	r7, 0, :.LBB22_20
// %bb.14:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB22_15:                              // %.preheader44
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_16 Depth 3
                                        //       Child Loop BB22_18 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB22_16
.LBB22_16:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB22_15 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB22_18
.LBB22_18:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB22_15 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, r12, :.LBB22_15
.LBB22_20:                              // %Flow5
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r6
.LBB22_21:                              // %Flow10
                                        //   in Loop: Header=BB22_1 Depth=1
	djmpeqoff	r6, 0, :.LBB22_35
// %bb.22:                              //   in Loop: Header=BB22_1 Depth=1
	dstcr	1, r7
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	0, r16, :.LBB22_30
// %bb.23:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB22_24:                              // %.preheader42
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_25 Depth 3
                                        //       Child Loop BB22_27 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB22_25
.LBB22_25:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB22_24 Depth=2
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB22_27
.LBB22_27:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB22_24 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, r12, :.LBB22_24
// %bb.29:                              // %Flow6
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r7
.LBB22_30:                              // %Flow8
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r6
	cp	crp2, crp3
	djmpeqoff	r7, 0, :.LBB22_35
// %bb.31:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB22_32:                              // %.preheader40
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_33 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB22_33
.LBB22_33:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB22_32 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r6, r12, :.LBB22_32
.LBB22_35:                              // %Flow11
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r6
.LBB22_36:                              // %Flow22
                                        //   in Loop: Header=BB22_1 Depth=1
	djmpeqoff	r6, 0, :.LBB22_62
// %bb.37:                              //   in Loop: Header=BB22_1 Depth=1
	dstcr	1, r5
	djmpeqoff	r17, 0, :.LBB22_52
// %bb.38:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	400, crp3
	dstcr	1, r6
	addi32	crp1, crp3, crp3
	dstcr	0, r5
	cp	crp3, crp4
	djmpeqoff	0, r16, :.LBB22_46
// %bb.39:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB22_40:                              // %.preheader38
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_41 Depth 3
                                        //       Child Loop BB22_43 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB22_41
.LBB22_41:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB22_40 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB22_43
.LBB22_43:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB22_40 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r5, r12, :.LBB22_40
// %bb.45:                              // %Flow12
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r6
.LBB22_46:                              // %Flow14
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r5
	djmpeqoff	r6, 0, :.LBB22_51
// %bb.47:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB22_48:                              // %.preheader36
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_49 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB22_49
.LBB22_49:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB22_48 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, r12, :.LBB22_48
.LBB22_51:                              // %Flow15
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r5
.LBB22_52:                              // %Flow20
                                        //   in Loop: Header=BB22_1 Depth=1
	djmpeqoff	r5, 0, :.LBB22_62
// %bb.53:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	400, crp3
	dstcr	1, r5
	addi32	crp1, crp3, crp3
	dstcr	0, r17
	cp	crp3, crp4
	djmpeqoff	0, r16, :.LBB22_59
// %bb.54:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB22_55:                              // %.preheader34
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_56 Depth 3
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB22_56
.LBB22_56:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB22_55 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r17, r12, :.LBB22_55
// %bb.58:                              // %Flow16
                                        //   in Loop: Header=BB22_1 Depth=1
	dstcr	0, r5
.LBB22_59:                              // %Flow18
                                        //   in Loop: Header=BB22_1 Depth=1
	djmpeqoff	r5, 0, :.LBB22_62
// %bb.60:                              //   in Loop: Header=BB22_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x180, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB22_61
.LBB22_61:                              // %.preheader
                                        //   Parent Loop BB22_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp3+=1]
.LBB22_62:                              // %.loopexit
                                        //   in Loop: Header=BB22_1 Depth=1
	stcr	400, crp3
	dshrlb	r15, 4, r15
	addi32	crp1, crp3, crp3
	addi32	crp1, 16, crp4          //      
	dstcr	0x200, pc.mode, south
	dstcr	0x180, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB22_63
.LBB22_63:                              //   Parent Loop BB22_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	stcr	0x2, bitwidthmode
	muli32lohi{10}	[crp3+=1], cr10, cr15
	stcr	0x0, bitwidthmode
	cmplti32	0, cr15, cr16
	csel	cr12, cr11, cr16
	addi32	cr16, cr15, cr15
	mini32	cr15, cr13, cr15
	maxi32	cr15, cr14, cr15
	shrlb.lb	cr15, 16, [crp4.z+=1]
// %bb.64:                              //   in Loop: Header=BB22_1 Depth=1
	dshlb	r13, 7, r13
	dshlb	r15, 6, r15
	daddi32	[rp2], r13, r13
	dcmplt32	26, r14, r14
	daddi32	r13, r15, r15
	dcsel	10, 16, r13
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r15, pls.addr, north
	dcp	r13, pls.count1, north
	dstcr	0, r14
	addi32	crp1, 16, crp3          //      
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	stcr	0x2, bitwidthmode
.LBB22_65:                              //   Parent Loop BB22_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB22_66 Depth 3
                                        //       Child Loop BB22_68 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB22_66
	dstcr	0x200, pc.mode, north
.LBB22_66:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.67:                              //   in Loop: Header=BB22_65 Depth=2
	dcp	r13, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB22_68
.LBB22_68:                              //   Parent Loop BB22_1 Depth=1
                                        //     Parent Loop BB22_65 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.69:                              //   in Loop: Header=BB22_65 Depth=2
	addi32	crp3, 4, crp3
	djmpincne	r14, 96, :.LBB22_65
// %bb.70:                              //   in Loop: Header=BB22_1 Depth=1
	dstcr	0x200, pc.mode, north
	djmpincne	r10, 4, :.LBB22_1
// %bb.71:
	dcp	[rp1 + 4], r8
	daddi32	rp1, 24, rp1
	stcr	1936, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj96ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj888832EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj26ELj26EEEEvRT0_RT1_RT2_
_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj96ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj888832EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj26ELj26EEEEvRT0_RT1_RT2_: // @_Z103fused_nn_conv2d_cast_fixed_point_multiply_clip_cast_multiply_nn_bias_add_cast_fi_2715872555147105862__3I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj96ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj888832EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj64ELj26ELj26EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -24, rp1
	stcr	-656, cr10
	stcr	272, crp2
	addi32	crp1, cr10, crp1        //     
	dcp	r12, rp2
	dcp	r11, rp3
	dcp	r10, rp4
	dstcr	0, r10
	dstcr	-1, r11
	addi32	crp1, crp2, crp2
	stcr	-384, crp3
	stcr	9057, cr10
	stcr	11626, cr11
	stcr	11359, cr12
	stcr	6553, cr13
	stcr	9680, cr14
	stcr	-32768, cr15
	stcr	32768, cr16
	stcr	8323072, cr17
	stcr	-8323072, cr5
	dstcr	256, r12
	dstcr	0x2, mode
	dcp	r8, [rp1 + 4]
	dstcr	0x20, pls.stride1, south
	dstcr	0x60, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x340, pls.stride2, south
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x1b200, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x20, pls.stride1, north
	dstcr	0x40, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x340, pls.stride2, north
.LBB23_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_5 Depth 2
                                        //       Child Loop BB23_6 Depth 3
                                        //       Child Loop BB23_8 Depth 3
                                        //       Child Loop BB23_10 Depth 3
                                        //     Child Loop BB23_15 Depth 2
                                        //       Child Loop BB23_16 Depth 3
                                        //       Child Loop BB23_18 Depth 3
                                        //     Child Loop BB23_24 Depth 2
                                        //       Child Loop BB23_25 Depth 3
                                        //       Child Loop BB23_27 Depth 3
                                        //     Child Loop BB23_32 Depth 2
                                        //       Child Loop BB23_33 Depth 3
                                        //     Child Loop BB23_40 Depth 2
                                        //       Child Loop BB23_41 Depth 3
                                        //       Child Loop BB23_43 Depth 3
                                        //     Child Loop BB23_48 Depth 2
                                        //       Child Loop BB23_49 Depth 3
                                        //     Child Loop BB23_55 Depth 2
                                        //       Child Loop BB23_56 Depth 3
                                        //     Child Loop BB23_61 Depth 2
                                        //     Child Loop BB23_63 Depth 2
                                        //       Child Loop BB23_64 Depth 3
                                        //       Child Loop BB23_66 Depth 3
                                        //     Child Loop BB23_69 Depth 2
                                        //       Child Loop BB23_70 Depth 3
                                        //       Child Loop BB23_72 Depth 3
	dshrlb	r10, 1, r14
	dshlb	r10, 4, r15
	dcmpeq32	r14, 0, r13
	dshlb	r14, 4, r13
	dandb	r15, 16, r15
	dcsel	r13, 4, r16
	dcmpneq32	r14, 0, r17
	dsubi32	r13, r16, r14
	dsubi32	10, r13, r5
	dcsel	r14, 0, r14
	dmaxi32	r5, 0, r7
	dsubi32	26, r15, r5
	dshlb	r14, 7, r14
	dmini32	r5, 20, r5
	daddi32	[rp4], r14, r14
	dshlb	r15, 2, r6
	dsubi32	60, r5, r28
	daddi32	r14, r6, r29
	dshrlb	r11, r28, r14
	dcmplti32	28, r5, r6
	dcsel	r14, 0, r28
	dsubi32	28, r5, r14
	dcmplti32	r5, 28, r5
	dshrlb	r11, r14, r5
	daddi32	r13, 16, r14
	dcsel	r5, -1, r5
	dcmpeq32	r15, 0, r30
	dandb	r5, -16, r6
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcsel	r6, r5, r30
	dcmplt32	r14, 26, r2
	daddi32	[rp4 + 3], r7, r7
	dsubi32	26, r13, r5
	dcmplt32	25, r14, r6
	dmuli32	r7, r2, r7
	dsubi32	4, r16, r16
	dmin32	r5, 16, r8
	dshlb	r6, 2, r5
	dcmpneq32	r17, 0, r17
	dxorb	r5, 20, r17
	dcsel	r16, 4, r5
	daddi32	r7, r8, r7
	dsubi32	4, r5, r16
	dmin32	r7, r17, r7
	dcp	r28, pls.maskh, south
	daddi32	r7, r16, r17
	dcp	r30, pls.maskl, south
	dcp	r29, pls.addr, south
	dcp	r17, pls.count1, south
	dstcr	1, r6
	dsubi32	20, r7, r16
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r5, 0, :.LBB23_36
// %bb.2:                               //   in Loop: Header=BB23_1 Depth=1
	dstcr	1, r6
	djmpeqoff	r17, 0, :.LBB23_21
// %bb.3:                               //   in Loop: Header=BB23_1 Depth=1
	dstcr	1, r7
	dstcr	0, r6
	cp	crp2, crp4
	djmpeqoff	0, r16, :.LBB23_13
// %bb.4:                               //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB23_5:                               // %.preheader46
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_6 Depth 3
                                        //       Child Loop BB23_8 Depth 3
                                        //       Child Loop BB23_10 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB23_6
.LBB23_6:                               //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB23_5 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB23_8
.LBB23_8:                               //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB23_5 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB23_10
.LBB23_10:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB23_5 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r6, 96, :.LBB23_5
// %bb.12:                              // %Flow
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r7
.LBB23_13:                              // %Flow5
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r6
	cp	crp2, crp4
	djmpeqoff	r7, 0, :.LBB23_20
// %bb.14:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB23_15:                              // %.preheader44
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_16 Depth 3
                                        //       Child Loop BB23_18 Depth 3
	dcp	r5, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB23_16
.LBB23_16:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB23_15 Depth=2
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB23_18
.LBB23_18:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB23_15 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r6, 96, :.LBB23_15
.LBB23_20:                              // %Flow6
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r6
.LBB23_21:                              // %Flow11
                                        //   in Loop: Header=BB23_1 Depth=1
	djmpeqoff	r6, 0, :.LBB23_35
// %bb.22:                              //   in Loop: Header=BB23_1 Depth=1
	dstcr	1, r7
	dstcr	0, r6
	cp	crp2, crp4
	djmpeqoff	0, r16, :.LBB23_30
// %bb.23:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB23_24:                              // %.preheader42
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_25 Depth 3
                                        //       Child Loop BB23_27 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB23_25
.LBB23_25:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB23_24 Depth=2
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB23_27
.LBB23_27:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_24 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB23_24 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r6, 96, :.LBB23_24
// %bb.29:                              // %Flow7
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r7
.LBB23_30:                              // %Flow9
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r6
	cp	crp2, crp4
	djmpeqoff	r7, 0, :.LBB23_35
// %bb.31:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB23_32:                              // %.preheader40
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_33 Depth 3
	dcp	r5, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB23_33
.LBB23_33:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_32 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB23_32 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r6, 96, :.LBB23_32
.LBB23_35:                              // %Flow12
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r6
.LBB23_36:                              // %Flow23
                                        //   in Loop: Header=BB23_1 Depth=1
	djmpeqoff	r6, 0, :.LBB23_62
// %bb.37:                              //   in Loop: Header=BB23_1 Depth=1
	dstcr	1, r5
	djmpeqoff	r17, 0, :.LBB23_52
// %bb.38:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	272, crp4
	dstcr	1, r6
	addi32	crp1, crp4, crp4
	dstcr	0, r5
	cp	crp4, crp5
	djmpeqoff	0, r16, :.LBB23_46
// %bb.39:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
.LBB23_40:                              // %.preheader38
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_41 Depth 3
                                        //       Child Loop BB23_43 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB23_41
.LBB23_41:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.42:                              //   in Loop: Header=BB23_40 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB23_43
.LBB23_43:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_40 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB23_40 Depth=2
	cp	south, [crp5+=1]
	djmpincne	r5, 96, :.LBB23_40
// %bb.45:                              // %Flow13
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r6
.LBB23_46:                              // %Flow15
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r5
	djmpeqoff	r6, 0, :.LBB23_51
// %bb.47:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x300, pc.mode, south
.LBB23_48:                              // %.preheader36
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_49 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB23_49
.LBB23_49:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_48 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.50:                              //   in Loop: Header=BB23_48 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r5, 96, :.LBB23_48
.LBB23_51:                              // %Flow16
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r5
.LBB23_52:                              // %Flow21
                                        //   in Loop: Header=BB23_1 Depth=1
	djmpeqoff	r5, 0, :.LBB23_62
// %bb.53:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	272, crp4
	dstcr	1, r5
	addi32	crp1, crp4, crp4
	dstcr	0, r17
	cp	crp4, crp5
	djmpeqoff	0, r16, :.LBB23_59
// %bb.54:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
	dstcr	0x200, pc.mode, south
.LBB23_55:                              // %.preheader34
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_56 Depth 3
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB23_56
.LBB23_56:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_55 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB23_55 Depth=2
	cp	south, [crp5+=1]
	djmpincne	r17, 96, :.LBB23_55
// %bb.58:                              // %Flow17
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0, r5
.LBB23_59:                              // %Flow19
                                        //   in Loop: Header=BB23_1 Depth=1
	djmpeqoff	r5, 0, :.LBB23_62
// %bb.60:                              //   in Loop: Header=BB23_1 Depth=1
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 96, :.LBB23_61
.LBB23_61:                              // %.preheader
                                        //   Parent Loop BB23_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp4+=1]
.LBB23_62:                              // %.loopexit
                                        //   in Loop: Header=BB23_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r17
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r17, pls.addr, west
	dshrlb	r15, 4, r15
	dstcr	0, r16
	addi32	crp1, 16, crp4          //      
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB23_63:                              //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_64 Depth 3
                                        //       Child Loop BB23_66 Depth 3
	cp	crp2, crp5
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 48, :.LBB23_64
	stcr	0x0, accumall
.LBB23_64:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrxi8.lb	[crp5+=2]
// %bb.65:                              //   in Loop: Header=BB23_63 Depth=2
	addi32	crp5, crp3, crp5
	stcr	0x1, bitwidthmode
	djmpincsetup	0, 191, :.LBB23_66
	nrb	[crp5.z+=1], north | south | east | west
.LBB23_66:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrni8		<>	nnbr	r90
	macwrni8.lb		<>	nrb	[crp5.z+=1], north | south | east | west
// %bb.67:                              //   in Loop: Header=BB23_63 Depth=2
	macwrni8		<>	nnbr	r90
	macwrni8	
	accsumsh8	0
	cp	accum0, cr6
	cp	accum0h, cr7
	stcr	0x2, bitwidthmode
	addi32	cr6, cr7, cr6
	muli32lohi{23}	cr6, cr10, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	muli32	>wl, cr6, cr6
	addi32	>wl, cr6, cr6
	stcr	0x0, bitwidthmode
	muli32lohi{19}	cr6, cr11, cr6
	mini32	cr6, 127, cr6
	maxi32	cr6, -127, cr6
	shlb	cr6, 16, cr6
	muli32lohi{17}	cr6, cr12, cr6
	muli32lohi{16}	cr6, cr13, cr7
	cmplti32	0, cr6, cr28
	csel	cr6, cr7, cr6
	muli32lohi{9}	cr6, cr14, cr6
	cmplti32	0, cr6, cr7
	csel	cr16, cr15, cr7
	addi32	cr7, cr6, cr6
	mini32	cr6, cr17, cr6
	maxi32	cr6, cr5, cr6
	shrlb	cr6, 16, [crp4.z+=1]
	djmpincne	r16, r12, :.LBB23_63
// %bb.68:                              //   in Loop: Header=BB23_1 Depth=1
	dshlb	r13, 7, r13
	dshlb	r15, 6, r15
	daddi32	[rp2], r13, r13
	dcmplt32	26, r14, r14
	daddi32	r13, r15, r15
	dcsel	10, 16, r13
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r15, pls.addr, north
	dcp	r13, pls.count1, north
	dstcr	0, r14
	addi32	crp1, 16, crp4          //      
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	stcr	0x2, bitwidthmode
.LBB23_69:                              //   Parent Loop BB23_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_70 Depth 3
                                        //       Child Loop BB23_72 Depth 3
	nrb	[crp4], north
	djmpincsetup	0, 4, :.LBB23_70
	dstcr	0x200, pc.mode, north
.LBB23_70:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.71:                              //   in Loop: Header=BB23_69 Depth=2
	dcp	r13, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB23_72
.LBB23_72:                              //   Parent Loop BB23_1 Depth=1
                                        //     Parent Loop BB23_69 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB23_69 Depth=2
	addi32	crp4, 4, crp4
	djmpincne	r14, 64, :.LBB23_69
// %bb.74:                              //   in Loop: Header=BB23_1 Depth=1
	dstcr	0x200, pc.mode, north
	djmpincne	r10, 4, :.LBB23_1
// %bb.75:
	dcp	[rp1 + 4], r8
	daddi32	rp1, 24, rp1
	stcr	656, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z104fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj67320EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj26ELj26EEEEvRT0_RT1_RT2_
_Z104fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj67320EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj26ELj26EEEEvRT0_RT1_RT2_: // @_Z104fused_nn_conv2d_nn_bias_add_cast_fixed_point_multiply_clip_cast_cast_cast_fixed__11151952881636687943__1I10FixedPointIsLh6ELh2ELi0EE7_TensorIDv4_aL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj64ELj26ELj26EEES2_IaLS4_0EjLj64ELS5_1EJLj1ELj1ELj1ELj67320EEES2_IS0_IiLh16ELh4ELi0EELS4_0EjLj64ELS5_1EJLj1ELj255ELj26ELj26EEEEvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -16, rp1
	stcr	-1296, cr10
	stcr	1040, crp2
	addi32	crp1, cr10, crp1        //     
	dcp	r12, rp2
	dcp	r11, rp3
	dcp	r10, rp4
	dstcr	0, r10
	dstcr	-1, r11
	addi32	crp1, crp2, crp2
	stcr	15990, cr10
	stcr	10769, cr11
	dstcr	0x20, pls.stride1, south
	dstcr	0x40, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x340, pls.stride2, south
	stcr	0x0, vapmode
	dstcr	0x0, pls.maskh, west
	dstcr	0xffffff, pls.maskl, west
	dstcr	0x10, pls.stride1, west
	dstcr	0x20df, pls.count1, west
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x20, pls.stride1, north
	dstcr	0x2, mode
	stcr	0x2, bitwidthmode
	dstcr	0xff, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x340, pls.stride2, north
.LBB24_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_63 Depth 2
                                        //       Child Loop BB24_64 Depth 3
                                        //       Child Loop BB24_66 Depth 3
                                        //       Child Loop BB24_68 Depth 3
                                        //     Child Loop BB24_5 Depth 2
                                        //       Child Loop BB24_6 Depth 3
                                        //       Child Loop BB24_8 Depth 3
                                        //     Child Loop BB24_14 Depth 2
                                        //       Child Loop BB24_15 Depth 3
                                        //       Child Loop BB24_17 Depth 3
                                        //     Child Loop BB24_22 Depth 2
                                        //       Child Loop BB24_23 Depth 3
                                        //     Child Loop BB24_57 Depth 2
                                        //       Child Loop BB24_58 Depth 3
                                        //       Child Loop BB24_60 Depth 3
                                        //     Child Loop BB24_31 Depth 2
                                        //       Child Loop BB24_32 Depth 3
                                        //     Child Loop BB24_38 Depth 2
                                        //       Child Loop BB24_39 Depth 3
                                        //     Child Loop BB24_44 Depth 2
                                        //     Child Loop BB24_46 Depth 2
                                        //       Child Loop BB24_47 Depth 3
                                        //     Child Loop BB24_50 Depth 2
                                        //       Child Loop BB24_51 Depth 3
                                        //       Child Loop BB24_53 Depth 3
	dshrlb	r10, 1, r13
	dshlb	r10, 4, r14
	dcmpeq32	r13, 0, r12
	dshlb	r13, 4, r12
	dandb	r14, 16, r14
	dcsel	r12, 4, r15
	dcmpneq32	r13, 0, r16
	dsubi32	r12, r15, r13
	dsubi32	10, r12, r17
	dcsel	r13, 0, r13
	dmaxi32	r17, 0, r6
	dsubi32	26, r14, r17
	dshlb	r13, 7, r13
	dmini32	r17, 20, r17
	daddi32	[rp4], r13, r13
	dshlb	r14, 2, r5
	dsubi32	60, r17, r7
	daddi32	r13, r5, r28
	dshrlb	r11, r7, r13
	dcmplti32	28, r17, r5
	dcsel	r13, 0, r7
	dsubi32	28, r17, r13
	dcmplti32	r17, 28, r17
	dshrlb	r11, r13, r17
	daddi32	r12, 16, r13
	dcsel	r17, -1, r17
	dcmpeq32	r14, 0, r29
	dandb	r17, -16, r5
	dstcr	0x71, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcsel	r5, r17, r29
	dcmplt32	r13, 26, r30
	daddi32	[rp4 + 3], r6, r6
	dsubi32	26, r12, r17
	dcmplt32	25, r13, r5
	dmuli32	r6, r30, r6
	dsubi32	4, r15, r15
	dmin32	r17, 16, r2
	dshlb	r5, 2, r17
	dcmpneq32	r16, 0, r16
	dxorb	r17, 20, r16
	dcsel	r15, 4, r17
	daddi32	r6, r2, r6
	dsubi32	4, r17, r15
	dmin32	r6, r16, r6
	dcp	r7, pls.maskh, south
	daddi32	r6, r15, r16
	dcp	r29, pls.maskl, south
	dcp	r28, pls.addr, south
	dcp	r16, pls.count1, south
	dstcr	1, r5
	dsubi32	20, r6, r15
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	djmpeqoff	r17, 0, :.LBB24_26
// %bb.2:                               //   in Loop: Header=BB24_1 Depth=1
	dstcr	1, r5
	djmpeqoff	r16, 0, :.LBB24_11
// %bb.3:                               //   in Loop: Header=BB24_1 Depth=1
	dstcr	1, r6
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	0, r15, :.LBB24_4
.LBB24_63:                              // %.preheader23
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_64 Depth 3
                                        //       Child Loop BB24_66 Depth 3
                                        //       Child Loop BB24_68 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB24_64
.LBB24_64:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB24_63 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB24_66
.LBB24_66:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.67:                              //   in Loop: Header=BB24_63 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB24_68
.LBB24_68:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_63 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.69:                              //   in Loop: Header=BB24_63 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 64, :.LBB24_63
// %bb.70:                              // %Flow
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r6
.LBB24_4:                               // %Flow5
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	r6, 0, :.LBB24_10
.LBB24_5:                               // %.preheader21
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_6 Depth 3
                                        //       Child Loop BB24_8 Depth 3
	dcp	r17, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB24_6
.LBB24_6:                               //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB24_5 Depth=2
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB24_8
.LBB24_8:                               //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB24_5 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 64, :.LBB24_5
.LBB24_10:                              // %Flow6
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r5
.LBB24_11:                              // %Flow11
                                        //   in Loop: Header=BB24_1 Depth=1
	djmpeqoff	r5, 0, :.LBB24_25
// %bb.12:                              //   in Loop: Header=BB24_1 Depth=1
	dstcr	1, r6
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	0, r15, :.LBB24_20
// %bb.13:                              //   in Loop: Header=BB24_1 Depth=1
	dstcr	0x200, pc.mode, south
.LBB24_14:                              // %.preheader19
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_15 Depth 3
                                        //       Child Loop BB24_17 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB24_15
.LBB24_15:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_14 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB24_14 Depth=2
	dcp	r15, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB24_17
.LBB24_17:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_14 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB24_14 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 64, :.LBB24_14
// %bb.19:                              // %Flow7
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r6
.LBB24_20:                              // %Flow9
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r5
	cp	crp2, crp3
	djmpeqoff	r6, 0, :.LBB24_25
// %bb.21:                              //   in Loop: Header=BB24_1 Depth=1
	dstcr	0x200, pc.mode, south
.LBB24_22:                              // %.preheader17
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_23 Depth 3
	dcp	r17, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB24_23
.LBB24_23:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_22 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB24_22 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r5, 64, :.LBB24_22
.LBB24_25:                              // %Flow12
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r5
.LBB24_26:                              // %Flow23
                                        //   in Loop: Header=BB24_1 Depth=1
	djmpeqoff	r5, 0, :.LBB24_45
// %bb.27:                              //   in Loop: Header=BB24_1 Depth=1
	dstcr	1, r17
	djmpeqoff	r16, 0, :.LBB24_35
// %bb.28:                              //   in Loop: Header=BB24_1 Depth=1
	stcr	1040, crp3
	dstcr	1, r5
	addi32	crp1, crp3, crp3
	dstcr	0, r17
	cp	crp3, crp4
	djmpeqoff	0, r15, :.LBB24_29
.LBB24_57:                              // %.preheader15
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_58 Depth 3
                                        //       Child Loop BB24_60 Depth 3
	dcp	r16, jumpendcount
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB24_58
.LBB24_58:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_57 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.59:                              //   in Loop: Header=BB24_57 Depth=2
	dcp	r15, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB24_60
.LBB24_60:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_57 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.61:                              //   in Loop: Header=BB24_57 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r17, 64, :.LBB24_57
// %bb.62:                              // %Flow13
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r5
.LBB24_29:                              // %Flow15
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r17
	djmpeqoff	r5, 0, :.LBB24_34
// %bb.30:                              //   in Loop: Header=BB24_1 Depth=1
	dstcr	0x300, pc.mode, south
.LBB24_31:                              // %.preheader13
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_32 Depth 3
	dcp	r16, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB24_32
.LBB24_32:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_31 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB24_31 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r17, 64, :.LBB24_31
.LBB24_34:                              // %Flow16
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r17
.LBB24_35:                              // %Flow21
                                        //   in Loop: Header=BB24_1 Depth=1
	djmpeqoff	r17, 0, :.LBB24_45
// %bb.36:                              //   in Loop: Header=BB24_1 Depth=1
	stcr	1040, crp3
	dstcr	1, r17
	addi32	crp1, crp3, crp3
	dstcr	0, r16
	cp	crp3, crp4
	djmpeqoff	0, r15, :.LBB24_42
// %bb.37:                              //   in Loop: Header=BB24_1 Depth=1
	dstcr	0x200, pc.mode, south
.LBB24_38:                              // %.preheader11
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_39 Depth 3
	dcp	r15, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB24_39
.LBB24_39:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_38 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.40:                              //   in Loop: Header=BB24_38 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r16, 64, :.LBB24_38
// %bb.41:                              // %Flow17
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0, r17
.LBB24_42:                              // %Flow19
                                        //   in Loop: Header=BB24_1 Depth=1
	djmpeqoff	r17, 0, :.LBB24_45
// %bb.43:                              //   in Loop: Header=BB24_1 Depth=1
	djmpincsetup	0, 64, :.LBB24_44
.LBB24_44:                              // %.preheader
                                        //   Parent Loop BB24_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp.lb	south, [crp3+=1]
.LBB24_45:                              // %.loopexit
                                        //   in Loop: Header=BB24_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r16
	dstcr	0x13, pls.mode, west
	dstcr	0x308, pc.mode, west
	dcp	r16, pls.addr, west
	dshrlb	r14, 4, r14
	dstcr	0, r15
	addi32	crp1, 16, crp3          //      
	dcp	[rp3 + 1], dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, [rp3 + 1]
.LBB24_46:                              //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_47 Depth 3
	cp	crp2, crp4
	djmpincsetup	0, 32, :.LBB24_47
	stcr	0x0, accumall
.LBB24_47:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_46 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	macwrxi8.lb	[crp4+=2]
// %bb.48:                              //   in Loop: Header=BB24_46 Depth=2
	accsumsh8	0
	cp	accum0, cr12
	cp	accum0h, cr13
	addi32	cr12, cr13, cr12
	addi32	>wl, cr12, cr12
	muli32lohi{23}	cr12, cr10, cr12
	mini32	cr12, 127, cr12
	maxi32	cr12, -127, cr12
	shlb	cr12, 16, cr12
	muli32lohi{16}	cr12, cr11, [crp3+=1]
	djmpincne	r15, 255, :.LBB24_46
// %bb.49:                              //   in Loop: Header=BB24_1 Depth=1
	dshlb	r12, 7, r12
	dshlb	r14, 6, r14
	daddi32	[rp2], r12, r12
	dcmplt32	26, r13, r13
	daddi32	r12, r14, r14
	dcsel	10, 16, r12
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r14, pls.addr, north
	dcp	r12, pls.count1, north
	dstcr	0, r13
	addi32	crp1, 16, crp3          //      
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
.LBB24_50:                              //   Parent Loop BB24_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_51 Depth 3
                                        //       Child Loop BB24_53 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB24_51
	dstcr	0x200, pc.mode, north
.LBB24_51:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_50 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.52:                              //   in Loop: Header=BB24_50 Depth=2
	dcp	r12, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB24_53
.LBB24_53:                              //   Parent Loop BB24_1 Depth=1
                                        //     Parent Loop BB24_50 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.54:                              //   in Loop: Header=BB24_50 Depth=2
	addi32	crp3, 4, crp3
	djmpincne	r13, 255, :.LBB24_50
// %bb.55:                              //   in Loop: Header=BB24_1 Depth=1
	dstcr	0x200, pc.mode, north
	djmpincne	r10, 4, :.LBB24_1
// %bb.56:
	daddi32	rp1, 16, rp1
	stcr	1296, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z16fused_reshape_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj26ELj26EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj676EEEEvRT0_RT1_
_Z16fused_reshape_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj26ELj26EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj676EEEEvRT0_RT1_: // @_Z16fused_reshape_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj26ELj26EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj255ELj1ELj676EEEEvRT0_RT1_
// %bb.0:
	daddi32	rp1, -32, rp1
	dstcr	0x2, mode
	dcp	r11, rp2
	dcp	r10, rp3
	dcp	r8, [rp1 + 7]
	dstcr	0, r10
	dcp	r9, [rp1 + 6]
	dstcr	65535, r11
	dcp	r18, [rp1 + 5]
	dstcr	-1431655765, r12
	dcp	r19, [rp1 + 4]
	dstcr	676, r13
	dcp	r20, [rp1 + 3]
	dstcr	65280, r14
	dcp	r21, [rp1 + 2]
	stcr	1626496491, cr11
	dcp	r22, [rp1 + 1]
	stcr	-676, cr12
	dcp	r23, [rp1]
	shlb	row, 4, cr10
	dstcr	33686018, r15
	addi32	cr10, col, cr10
	stcr	1321528399, cr13
	stcr	832, cr14
	dstcr	256, r16
	dstcr	691, r17
	dstcr	2752, r5
	dstcr	421, r6
	dstcr	677, r7
	dstcr	765, r28
                                        // implicit-def: $cx15
	dstcr	0x1, pls.count2, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	stcr	0x2, bitwidthmode
	dstcr	0x10, pls.stride1, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x2b0, pls.stride2, north
.LBB25_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB25_3 Depth 2
                                        //     Child Loop BB25_5 Depth 2
                                        //     Child Loop BB25_7 Depth 2
                                        //     Child Loop BB25_9 Depth 2
                                        //     Child Loop BB25_11 Depth 2
                                        //     Child Loop BB25_14 Depth 2
                                        //       Child Loop BB25_15 Depth 3
                                        //       Child Loop BB25_17 Depth 3
                                        //     Child Loop BB25_32 Depth 2
                                        //     Child Loop BB25_34 Depth 2
                                        //     Child Loop BB25_25 Depth 2
	dandb	r10, r11, r29
	dcpc	r13, cr16
	dmul32hi	r29, r12, r29
	dcp	[rp3], pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dshrlb	r29, 1, r29
	dstcr	0x10, pls.count1, north
	dmuli32	r29, -3, r2
	dmuli32	r29, r13, r30
	daddi32	r2, r10, r2
	dcpc	r29, cr17
	dcpc	r30, cr5
	dshlb	r2, 8, r30
	dandb	r30, r14, r30
	dcpc	r30, cr6
	addi32	cr10, cr6, cr6
	addi32	cr6, cr5, cr5
	cmplti32	cr6, cr16, cr6
	muli32hi	cr5, cr11, cr16
	shrlb	cr16, 31, cr7
	shrab	cr16, 8, cr16
	addi32	cr16, cr7, cr16
	muli32	cr16, cr12, cr7
	cmpltei32	cr16, cr17, cr28
	addi32	cr7, cr5, cr17
	andb	cr6, cr28, cr5
	predpush	cr5, :.LBB25_13
// %bb.2:                               //   in Loop: Header=BB25_1 Depth=1
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	[rp3 + 1], dependencyid
.LBB25_3:                               //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dandb	r15, plsstatus, r8
	dorb	r8, pelsr, r8
	djmpneqoff	r8, 0, :.LBB25_3
// %bb.4:                               //   in Loop: Header=BB25_1 Depth=1
	muli32hi	cr17, cr13, cr15
	muli32	cr16, cr14, cr16
	shrlb	cr15, 31, cr5
	shrab	cr15, 3, cr15
	addi32	cr17, cr16, cr16
	addi32	cr15, cr5, cr15
	djmpincsetup	0, 4, :.LBB25_5
	muli32	cr15, 6, cr15
	dstcr	0x1, plsstatus, north
	addi32	cr16, cr15, cr15
	shlb	cr15, 2, cr15
	nrb	cr15, north
	dstcr	0x260, pc.mode, north
.LBB25_5:                               //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB25_1 Depth=1
	djmpincsetup	0, 16, :.LBB25_7
	dstcr	0x360, pc.mode, north
.LBB25_7:                               //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB25_1 Depth=1
	djmpincsetup	0, 16, :.LBB25_9
.LBB25_9:                               // %.preheader8
                                        //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.10:                              //   in Loop: Header=BB25_1 Depth=1
	djmpincsetup	0, 4, :.LBB25_11
	dstcr	0x260, pc.mode, north
.LBB25_11:                              //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.12:                              //   in Loop: Header=BB25_1 Depth=1
	cp	north, cr15
.LBB25_13:                              // %Flow6
                                        //   in Loop: Header=BB25_1 Depth=1
	predpop	
	dandb	r2, r11, r8
	dsubi32	r13, r30, r2
	dcmplt32	r8, 2, r9
	dshrlb	r2, 8, r9
	dcsel	0, 164, r2
	dcsel	1, r9, r18
	dcp	[rp2], r9
	shlb	row, 4, cr16
	dstcr	0, r19
	addi32	cr16, col, cr16
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	djmpeqoff	r18, 0, :.LBB25_19
.LBB25_14:                              // %.preheader
                                        //   Parent Loop BB25_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB25_15 Depth 3
                                        //       Child Loop BB25_17 Depth 3
	dshlb	r19, 8, r20
	dmuli32	r29, r5, r21
	daddi32	r20, r30, r20
	djmpincsetup	0, 4, :.LBB25_15
	dshrab	r20, 31, r22
	dsubi32	r17, r20, r23
	dshrlb	r22, 28, r22
	daddi32	r9, r21, r21
	daddi32	r20, r22, r22
	dcmplti32	r20, r6, r20
	dshlb	r22, 2, r20
	dshrab	r23, 31, r22
	dandb	r20, -64, r20
	dshrlb	r22, 28, r22
	daddi32	r21, r20, r20
	daddi32	r23, r22, r21
	dcp	r20, pls.addr, north
	dshrab	r21, 4, r21
	dcsel	16, r21, r20
	dcp	r20, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	cr15, north
	dstcr	0x200, pc.mode, north
.LBB25_15:                              //   Parent Loop BB25_1 Depth=1
                                        //     Parent Loop BB25_14 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB25_14 Depth=2
	djmpincsetup	0, 16, :.LBB25_17
	dstcr	0x300, pc.mode, north
.LBB25_17:                              //   Parent Loop BB25_1 Depth=1
                                        //     Parent Loop BB25_14 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB25_14 Depth=2
	djmpincne	r19, r18, :.LBB25_14
.LBB25_19:                              // %.loopexit7
                                        //   in Loop: Header=BB25_1 Depth=1
	djmplt	r8, 2, :.LBB25_27
// %bb.20:                              //   in Loop: Header=BB25_1 Depth=1
	dshlb	r18, 8, r8
	dmuli32	r29, r5, r18
	daddi32	r8, r30, r30
	dstcr	1, r29
	dshrab	r30, 31, r8
	daddi32	r9, r18, r9
	dshrlb	r8, 28, r8
	dsubi32	r17, r30, r18
	daddi32	r30, r8, r8
	daddi32	r30, r16, r19
	dshrab	r8, 4, r8
	dcmplti32	r30, r6, r30
	dshlb	r8, 6, r30
	dshrab	r18, 31, r8
	daddi32	r9, r30, r30
	dshrlb	r8, 28, r8
	dcp	r30, pls.addr, north
	daddi32	r18, r8, r8
	dshrab	r8, 4, r30
	dcsel	1, r30, r8
	dcmplt32	r19, r7, r9
	dcp	r8, pls.count1, north
	dcsel	1, r30, r30
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r30, :.LBB25_21
// %bb.29:                              //   in Loop: Header=BB25_1 Depth=1
	dcpc	r2, cr17
	cmplti32	cr16, cr17, cr17
	predpush	cr17, :.LBB25_31
// %bb.30:                              //   in Loop: Header=BB25_1 Depth=1
	nrb	cr15, north
.LBB25_31:                              //   in Loop: Header=BB25_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB25_32
	dstcr	0x200, pc.mode, north
.LBB25_32:                              //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB25_1 Depth=1
	dcp	r30, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB25_34
.LBB25_34:                              //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.35:                              // %Flow
                                        //   in Loop: Header=BB25_1 Depth=1
	dstcr	0, r29
.LBB25_21:                              // %Flow3
                                        //   in Loop: Header=BB25_1 Depth=1
	djmpeqoff	0, r29, :.LBB25_27
// %bb.22:                              //   in Loop: Header=BB25_1 Depth=1
	dcpc	r2, cr17
	cmplti32	cr16, cr17, cr16
	predpush	cr16, :.LBB25_24
// %bb.23:                              //   in Loop: Header=BB25_1 Depth=1
	nrb	cr15, north
.LBB25_24:                              //   in Loop: Header=BB25_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB25_25
	dstcr	0x200, pc.mode, north
.LBB25_25:                              //   Parent Loop BB25_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB25_1 Depth=1
	dstcr	0x300, pc.mode, north
.LBB25_27:                              // %.loopexit
                                        //   in Loop: Header=BB25_1 Depth=1
	dstcr	0x200, pc.mode, north
	djmpincne	r10, r28, :.LBB25_1
// %bb.28:
	dcp	[rp1], r23
	dcp	[rp1 + 1], r22
	dcp	[rp1 + 2], r21
	dcp	[rp1 + 3], r20
	dcp	[rp1 + 4], r19
	dcp	[rp1 + 5], r18
	dcp	[rp1 + 6], r9
	dcp	[rp1 + 7], r8
	daddi32	rp1, 32, rp1
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z18fused_transpose_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj676EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj676ELj1ELj255EEEEvRT0_RT1_
_Z18fused_transpose_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj676EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj676ELj1ELj255EEEEvRT0_RT1_: // @_Z18fused_transpose_16I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj255ELj1ELj676EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj676ELj1ELj255EEEEvRT0_RT1_
// %bb.0:
	addi32	crp1, -64, crp1         //     
	cp	row, cr10
	dcp	r11, rp2
	dcp	r10, rp3
	cp	col, cr11
	dstcr	0, r10
	cp	crp1, crp2
	dstcr	268435440, r11
	stcr	688, cr12
	dstcr	676, r12
	cmpeq32	cr10, 0, cr13
	dstcr	33686018, r13
	dstcr	10815, r14
	dstcr	0, r15
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	dstcr	0x2, mode
	stcr	0x2, bitwidthmode
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x100, pls.stride1, north
	dstcr	0x10, pls.stride2, north
.LBB26_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB26_2 Depth 2
                                        //       Child Loop BB26_4 Depth 3
                                        //       Child Loop BB26_6 Depth 3
                                        //       Child Loop BB26_8 Depth 3
                                        //       Child Loop BB26_10 Depth 3
                                        //       Child Loop BB26_12 Depth 3
                                        //     Child Loop BB26_16 Depth 2
                                        //       Child Loop BB26_17 Depth 3
	cp	crp2, crp3
	dstcr	0, r16
	dcp	[rp3], pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB26_2:                               //   Parent Loop BB26_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB26_4 Depth 3
                                        //       Child Loop BB26_6 Depth 3
                                        //       Child Loop BB26_8 Depth 3
                                        //       Child Loop BB26_10 Depth 3
                                        //       Child Loop BB26_12 Depth 3
	dshrab	r10, 31, r17
	stcr	0, cr14
	dshrlb	r17, 28, r17
	daddi32	r10, r17, r17
	dandb	r17, r11, r5
	dshrab	r17, 4, r17
	dsubi32	r10, r5, r5
	dcmplt32	r17, r12, r6
	dshlb	r5, 4, r5
	dcpc	r17, cr15
	dcpc	r6, cr16
	dcpc	r5, cr17
	addi32	cr17, cr11, cr17
	addi32	cr17, cr10, cr5
	cmplt32	cr17, 255, cr17
	muli32	cr5, cr12, cr5
	andb	cr13, cr17, cr17
	addi32	cr5, cr15, cr15
	andb	cr16, cr17, cr16
	predpush	cr16, :.LBB26_14
// %bb.3:                               //   in Loop: Header=BB26_2 Depth=2
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	[rp3 + 1], dependencyid
.LBB26_4:                               //   Parent Loop BB26_1 Depth=1
                                        //     Parent Loop BB26_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	dandb	r13, plsstatus, r17
	dorb	r17, pelsr, r17
	djmpneqoff	r17, 0, :.LBB26_4
// %bb.5:                               //   in Loop: Header=BB26_2 Depth=2
	shlb	cr15, 2, cr14
	djmpincsetup	0, 4, :.LBB26_6
	dstcr	0x1, plsstatus, north
	nrb	cr14, north
	dstcr	0x260, pc.mode, north
.LBB26_6:                               //   Parent Loop BB26_1 Depth=1
                                        //     Parent Loop BB26_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB26_2 Depth=2
	djmpincsetup	0, 16, :.LBB26_8
	dstcr	0x360, pc.mode, north
.LBB26_8:                               //   Parent Loop BB26_1 Depth=1
                                        //     Parent Loop BB26_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB26_2 Depth=2
	djmpincsetup	0, 16, :.LBB26_10
.LBB26_10:                              // %.preheader
                                        //   Parent Loop BB26_1 Depth=1
                                        //     Parent Loop BB26_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.11:                              //   in Loop: Header=BB26_2 Depth=2
	djmpincsetup	0, 4, :.LBB26_12
	dstcr	0x260, pc.mode, north
.LBB26_12:                              //   Parent Loop BB26_1 Depth=1
                                        //     Parent Loop BB26_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.13:                              //   in Loop: Header=BB26_2 Depth=2
	cp	north, cr14
.LBB26_14:                              // %Flow
                                        //   in Loop: Header=BB26_2 Depth=2
	predpop	
	daddi32	r10, 1, r10
	cp	cr14, [crp3+=1]
	djmpincne	r16, 16, :.LBB26_2
// %bb.15:                              //   in Loop: Header=BB26_1 Depth=1
	dshlb	r15, 6, r16
	cp	crp2, crp3
	daddi32	[rp2], r16, r5
	dstcr	0, r16
	dcp	[rp2 + 1], r17
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r5, pls.addr, north
	dstcr	0x1, pls.count1, north
	dstcr	0x10, pls.count2, north
	dcp	r17, dependencyid
	dstcr	0x1, plsstatus, north
.LBB26_16:                              //   Parent Loop BB26_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB26_17 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB26_17
	dstcr	0x200, pc.mode, north
.LBB26_17:                              //   Parent Loop BB26_1 Depth=1
                                        //     Parent Loop BB26_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB26_16 Depth=2
	addi32	crp3, 4, crp3
	dstcr	0x300, pc.mode, north
	nnb	south, north
	djmpincne	r16, 16, :.LBB26_16
// %bb.19:                              //   in Loop: Header=BB26_1 Depth=1
	daddi32	r15, 16, r15
	dstcr	0x200, pc.mode, north
	djmplte	r15, r14, :.LBB26_1
// %bb.20:
	addi32	crp1, 64, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z16fused_reshape_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj676ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj2028ELj1ELj85EEEEvRT0_RT1_
_Z16fused_reshape_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj676ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj2028ELj1ELj85EEEEvRT0_RT1_: // @_Z16fused_reshape_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj676ELj1ELj255EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj2028ELj1ELj85EEEEvRT0_RT1_
// %bb.0:
	dcp	r11, rp2
	dstcr	0x2, mode
	dcp	r10, rp3
	dstcr	0, r10
	dcp	[rp2], pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	shlb	row, 4, cr11
	stcr	1616928865, cr10
	addi32	cr11, col, cr11
	dstcr	676, r11
	cmplti32	cr11, 255, cr12
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x10, pls.count1, south
	dstcr	0x1, pls.count2, south
	stcr	0x2, bitwidthmode
	dstcr	0x0, plsthresholdsouth
	dstcr	0x100, pls.stride2, south
	dstcr	0x0, plsthresholdnorth
.LBB27_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_2 Depth 2
                                        //     Child Loop BB27_4 Depth 2
                                        //     Child Loop BB27_9 Depth 2
                                        //     Child Loop BB27_11 Depth 2
                                        //     Child Loop BB27_13 Depth 2
                                        //     Child Loop BB27_15 Depth 2
	dshlb	r10, 10, r12
	djmpincsetup	0, 16, :.LBB27_2
	daddi32	[rp3], r12, r12
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	cp	row, cr13
	cp	col, cr14
	dstcr	0x0, pc.constant, south
	dcp	r12, pls.addr, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB27_2:                               //   Parent Loop BB27_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.3:                               //   in Loop: Header=BB27_1 Depth=1
	djmpincsetup	0, 4, :.LBB27_4
	dstcr	0x200, pc.mode, south
.LBB27_4:                               //   Parent Loop BB27_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.5:                               //   in Loop: Header=BB27_1 Depth=1
	shlb	cr13, 4, cr15
	stcr	0, cr13
	addi32	cr15, cr14, cr14
	dstcr	0x200, pc.mode, south
	cmplti32	cr14, 255, cr14
	predpush	cr14, :.LBB27_7
// %bb.6:                               //   in Loop: Header=BB27_1 Depth=1
	cp	south, cr13
.LBB27_7:                               //   in Loop: Header=BB27_1 Depth=1
	predpop	
	dmuli32	r10, 255, r12
	dstcr	0x200, pc.mode, south
	dcpc	r12, cr14
	addi32	cr14, cr11, cr14
	predpush	cr12, :.LBB27_17
// %bb.8:                               //   in Loop: Header=BB27_1 Depth=1
	muli32hi	cr14, cr10, cr15
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	shrlb	cr15, 31, cr16
	shrab	cr15, 5, cr15
	dstcr	0x360, pc.mode, north
	addi32	cr15, cr16, cr15
	dcp	[rp2 + 1], dependencyid
	muli32	cr15, 11, cr15
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	addi32	cr15, cr14, cr14
	djmpincsetup	0, 4, :.LBB27_9
	shlb	cr14, 2, cr14
	dstcr	0x260, pc.mode, north
	nrb	cr14, north
.LBB27_9:                               //   Parent Loop BB27_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.10:                              //   in Loop: Header=BB27_1 Depth=1
	djmpincsetup	0, 16, :.LBB27_11
	dstcr	0x360, pc.mode, north
.LBB27_11:                              //   Parent Loop BB27_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.12:                              //   in Loop: Header=BB27_1 Depth=1
	djmpincsetup	0, 4, :.LBB27_13
	dstcr	0x260, pc.mode, north
	nrb	cr13, north
.LBB27_13:                              //   Parent Loop BB27_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB27_1 Depth=1
	djmpincsetup	0, 16, :.LBB27_15
	dstcr	0x360, pc.mode, north
.LBB27_15:                              //   Parent Loop BB27_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB27_1 Depth=1
	dstcr	0x260, pc.mode, north
.LBB27_17:                              // %Flow
                                        //   in Loop: Header=BB27_1 Depth=1
	predpop	
	djmpincne	r10, r11, :.LBB27_1
// %bb.18:
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z18fused_transpose_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2028ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj2028EEEEvRT0_RT1_
_Z18fused_transpose_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2028ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj2028EEEEvRT0_RT1_: // @_Z18fused_transpose_15I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2028ELj1ELj85EEES2_IS3_LS4_0EjLj64ELS5_1EJLj1ELj85ELj1ELj2028EEEEvRT0_RT1_
// %bb.0:
	daddi32	rp1, -8, rp1
	stcr	-520, cr10
	dcp	r11, rp2
	addi32	crp1, cr10, crp1        //     
	cp	row, cr10
	dcp	r10, rp3
	cp	col, cr11
	dstcr	0, r10
	addi32	crp1, 12, crp2          //      
	dstcr	-2130574327, r11
	stcr	2028, cr12
	cmpeq32	cr10, 0, cr13
	dstcr	33686018, r12
	dstcr	33818641, r13
	dstcr	8128, r14
	dstcr	10794, r15
	dstcr	0, r16
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	dstcr	0x2, mode
	stcr	0x2, bitwidthmode
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x7f0, pls.stride1, north
	dstcr	0x10, pls.stride2, north
.LBB28_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB28_2 Depth 2
                                        //       Child Loop BB28_4 Depth 3
                                        //       Child Loop BB28_6 Depth 3
                                        //       Child Loop BB28_8 Depth 3
                                        //       Child Loop BB28_10 Depth 3
                                        //       Child Loop BB28_12 Depth 3
                                        //     Child Loop BB28_16 Depth 2
                                        //       Child Loop BB28_17 Depth 3
	cp	crp2, crp3
	dstcr	0, r17
	dcp	[rp3], pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB28_2:                               //   Parent Loop BB28_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB28_4 Depth 3
                                        //       Child Loop BB28_6 Depth 3
                                        //       Child Loop BB28_8 Depth 3
                                        //       Child Loop BB28_10 Depth 3
                                        //       Child Loop BB28_12 Depth 3
	dmuli32hi	r10, r11, r5
	stcr	0, cr14
	daddi32	r5, r10, r5
	dshrlb	r5, 31, r6
	dshrab	r5, 6, r5
	daddi32	r5, r6, r5
	dmuli32	r5, -127, r6
	dcmplt32	r5, 85, r7
	dcpc	r5, cr15
	daddi32	r6, r10, r5
	dcpc	r7, cr16
	dshlb	r5, 4, r5
	dcpc	r5, cr17
	addi32	cr17, cr11, cr17
	addi32	cr17, cr10, cr5
	cmplt32	cr17, cr12, cr17
	muli32	cr5, 96, cr5
	andb	cr13, cr17, cr17
	addi32	cr5, cr15, cr15
	andb	cr16, cr17, cr16
	predpush	cr16, :.LBB28_14
// %bb.3:                               //   in Loop: Header=BB28_2 Depth=2
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	[rp3 + 1], dependencyid
.LBB28_4:                               //   Parent Loop BB28_1 Depth=1
                                        //     Parent Loop BB28_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	dandb	r12, plsstatus, r5
	dorb	r5, pelsr, r5
	djmpneqoff	r5, 0, :.LBB28_4
// %bb.5:                               //   in Loop: Header=BB28_2 Depth=2
	shlb	cr15, 2, cr14
	djmpincsetup	0, 4, :.LBB28_6
	dstcr	0x1, plsstatus, north
	nrb	cr14, north
	dstcr	0x260, pc.mode, north
.LBB28_6:                               //   Parent Loop BB28_1 Depth=1
                                        //     Parent Loop BB28_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB28_2 Depth=2
	djmpincsetup	0, 16, :.LBB28_8
	dstcr	0x360, pc.mode, north
.LBB28_8:                               //   Parent Loop BB28_1 Depth=1
                                        //     Parent Loop BB28_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB28_2 Depth=2
	djmpincsetup	0, 16, :.LBB28_10
.LBB28_10:                              // %.preheader
                                        //   Parent Loop BB28_1 Depth=1
                                        //     Parent Loop BB28_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.11:                              //   in Loop: Header=BB28_2 Depth=2
	djmpincsetup	0, 4, :.LBB28_12
	dstcr	0x260, pc.mode, north
.LBB28_12:                              //   Parent Loop BB28_1 Depth=1
                                        //     Parent Loop BB28_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	north, south
// %bb.13:                              //   in Loop: Header=BB28_2 Depth=2
	cp	north, cr14
.LBB28_14:                              // %Flow
                                        //   in Loop: Header=BB28_2 Depth=2
	predpop	
	daddi32	r10, 1, r10
	cp	cr14, [crp3+=1]
	djmpincne	r17, 127, :.LBB28_2
// %bb.15:                              //   in Loop: Header=BB28_1 Depth=1
	dmul32hi	r16, r13, r5
	cp	crp2, crp3
	dstcr	0, r17
	dsubi32	r16, r5, r7
	dcp	[rp2 + 1], r6
	dshrlb	r7, 1, r7
	daddi32	r7, r5, r5
	dshrlb	r5, 6, r5
	dmuli32	r5, r14, r5
	daddi32	[rp2], r5, r5
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r5, pls.addr, north
	dstcr	0x1, pls.count1, north
	dstcr	0x7f, pls.count2, north
	dcp	r6, dependencyid
	dstcr	0x1, plsstatus, north
.LBB28_16:                              //   Parent Loop BB28_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB28_17 Depth 3
	nrb	[crp3], north
	djmpincsetup	0, 4, :.LBB28_17
	dstcr	0x200, pc.mode, north
.LBB28_17:                              //   Parent Loop BB28_1 Depth=1
                                        //     Parent Loop BB28_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB28_16 Depth=2
	addi32	crp3, 4, crp3
	dstcr	0x300, pc.mode, north
	nnb	south, north
	djmpincne	r17, 127, :.LBB28_16
// %bb.19:                              //   in Loop: Header=BB28_1 Depth=1
	daddi32	r16, 127, r16
	dstcr	0x200, pc.mode, north
	djmplte	r16, r15, :.LBB28_1
// %bb.20:
	daddi32	rp1, 8, rp1
	stcr	520, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z50fused_sigmoid_fixed_point_multiply_cast_cast_add_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_
_Z50fused_sigmoid_fixed_point_multiply_cast_cast_add_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_: // @_Z50fused_sigmoid_fixed_point_multiply_cast_cast_add_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -80, rp1
	daddi32	rp1, 72, rp2
	dstcr	0x2, mode
	addi32	crp1, -72, crp1         //     
	dcp	r11, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 64, rp2
	dcp	r10, rp4
	dcp	r9, [rp2]
	daddi32	rp1, 56, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 48, rp2
	dstcr	256, r11
	dcp	r19, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	2043, r13
	dcp	r20, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	1773, r14
	dcp	r21, [rp2]
	dcp	r12, rp2
	dstcr	2028, r12
	addi32	crp1, 24, crp2          //      
	dstcr	8128, r15
	cp	crp1, crp3
	stcr	-65536, cr10
	stcr	65536, cr11
	stcr	16384, cr12
	dstcr	1772, r16
	dstcr	2029, r17
	dcp	r22, [rp1 + 6]
	dcp	r23, [rp1 + 4]
	dcp	r24, [rp1 + 2]
	dcp	r25, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x7f0, pls.stride2, north
.LBB29_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB29_2 Depth 2
                                        //       Child Loop BB29_4 Depth 3
                                        //         Child Loop BB29_5 Depth 4
                                        //         Child Loop BB29_7 Depth 4
                                        //       Child Loop BB29_13 Depth 3
                                        //       Child Loop BB29_15 Depth 3
                                        //       Child Loop BB29_17 Depth 3
                                        //       Child Loop BB29_23 Depth 3
                                        //       Child Loop BB29_25 Depth 3
                                        //       Child Loop BB29_32 Depth 3
                                        //       Child Loop BB29_34 Depth 3
                                        //     Child Loop BB29_41 Depth 2
                                        //       Child Loop BB29_43 Depth 3
                                        //         Child Loop BB29_44 Depth 4
                                        //         Child Loop BB29_46 Depth 4
                                        //       Child Loop BB29_52 Depth 3
                                        //       Child Loop BB29_54 Depth 3
                                        //       Child Loop BB29_56 Depth 3
                                        //       Child Loop BB29_62 Depth 3
                                        //       Child Loop BB29_64 Depth 3
                                        //       Child Loop BB29_71 Depth 3
                                        //       Child Loop BB29_73 Depth 3
                                        //     Child Loop BB29_80 Depth 2
                                        //     Child Loop BB29_82 Depth 2
                                        //       Child Loop BB29_84 Depth 3
                                        //         Child Loop BB29_85 Depth 4
                                        //         Child Loop BB29_87 Depth 4
                                        //       Child Loop BB29_103 Depth 3
                                        //       Child Loop BB29_105 Depth 3
                                        //       Child Loop BB29_95 Depth 3
	dcmplt32	6, r10, r5
	dshlb	r10, 8, r5
	dcsel	r12, r11, r28
	dsubi32	r12, r5, r6
	dcmpneq32	r10, 7, r7
	dshrlb	r6, 8, r6
	dcp	[rp4], r29
	dandb	r6, 255, r8
	dstcr	0x11, pls.mode, south
	dcsel	1, r8, r8
	dstcr	0x300, pc.mode, south
	dshlb	r8, 8, r9
	dstcr	0x200, pc.mode, north
	daddi32	r9, r5, r20
	shlb	row, 4, cr13
	dsubi32	r28, r20, r9
	daddi32	r20, r11, r18
	daddi32	r9, 15, r19
	dandb	r9, 12, r22
	dshrab	r19, 31, r21
	dstcr	0, r30
	dshrlb	r21, 28, r21
	dstcr	0, r2
	daddi32	r19, r21, r19
	dcmpeq32	r22, 0, r21
	dandb	r19, -16, r19
	dsubi32	r13, r20, r21
	dcsel	r9, r19, r9
	dcmplt32	r28, r18, r18
	dshrab	r9, 31, r18
	dshrab	r21, 31, r19
	dshrlb	r18, 28, r18
	dshrlb	r19, 28, r19
	daddi32	r9, r18, r9
	daddi32	r21, r19, r18
	dshrab	r9, 4, r9
	dshrab	r18, 4, r19
	dcsel	r9, 16, r9
	dcmplt32	r20, r14, r18
	dcsel	1, r19, r19
	dcmpneq32	r7, 0, r7
	addi32	cr13, col, cr13
	dsubi32	16, r9, r18
	dshrlb	r20, 4, r20
	dcsel	0, 236, r7
	stcr	0x2, bitwidthmode
	dstcr	0x0, pc.constant, south
	dstcr	0x7f0, pls.stride2, south
.LBB29_2:                               //   Parent Loop BB29_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB29_4 Depth 3
                                        //         Child Loop BB29_5 Depth 4
                                        //         Child Loop BB29_7 Depth 4
                                        //       Child Loop BB29_13 Depth 3
                                        //       Child Loop BB29_15 Depth 3
                                        //       Child Loop BB29_17 Depth 3
                                        //       Child Loop BB29_23 Depth 3
                                        //       Child Loop BB29_25 Depth 3
                                        //       Child Loop BB29_32 Depth 3
                                        //       Child Loop BB29_34 Depth 3
	djmpeqoff	0, r8, :.LBB29_9
// %bb.3:                               //   in Loop: Header=BB29_2 Depth=2
	dshlb	r2, 2, r22
	dstcr	0, r21
	dcpc	r22, crp4
	addi32	crp2, crp4, crp4
.LBB29_4:                               // %.preheader17
                                        //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB29_5 Depth 4
                                        //         Child Loop BB29_7 Depth 4
	dshlb	r21, 8, r23
	dmuli32	r30, r15, r22
	daddi32	r23, r5, r23
	djmpincsetup	0, 16, :.LBB29_5
	daddi32	r29, r22, r22
	dshlb	r23, 2, r24
	dsubi32	r13, r23, r25
	daddi32	r22, r24, r22
	dshrab	r25, 31, r24
	dcmplt32	r23, r14, r23
	dshrlb	r24, 28, r23
	dcp	r22, pls.addr, south
	daddi32	r25, r23, r22
	dshrab	r22, 4, r22
	dcsel	16, r22, r22
	dcp	r22, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
.LBB29_5:                               //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        //       Parent Loop BB29_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB29_4 Depth=3
	djmpincsetup	0, 4, :.LBB29_7
	dstcr	0x200, pc.mode, south
.LBB29_7:                               //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        //       Parent Loop BB29_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB29_4 Depth=3
	cp	south, [crp4+=1]
	daddi32	r2, 1, r2
	djmpincne	r21, r8, :.LBB29_4
.LBB29_9:                               // %.loopexit18
                                        //   in Loop: Header=BB29_2 Depth=2
	djmpneqoff	7, r10, :.LBB29_39
// %bb.10:                              //   in Loop: Header=BB29_2 Depth=2
	dmuli32	r30, r15, r22
	dshlb	r20, 6, r23
	dstcr	1, r21
	daddi32	r29, r22, r22
                                        // implicit-def: $cx14
	daddi32	r22, r23, r22
	dcp	r22, pls.addr, south
	dcp	r19, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r9, 0, :.LBB29_30
// %bb.11:                              //   in Loop: Header=BB29_2 Depth=2
	dstcr	1, r21
                                        // implicit-def: $cx14
	djmpeqoff	0, r18, :.LBB29_21
// %bb.12:                              //   in Loop: Header=BB29_2 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB29_13
.LBB29_13:                              // %.preheader16
                                        //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB29_2 Depth=2
	djmpincsetup	0, 4, :.LBB29_15
	dstcr	0x200, pc.mode, south
.LBB29_15:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB29_2 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB29_17
.LBB29_17:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB29_2 Depth=2
	dcpc	r7, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB29_20
// %bb.19:                              //   in Loop: Header=BB29_2 Depth=2
	cp	south, cr14
.LBB29_20:                              // %Flow18
                                        //   in Loop: Header=BB29_2 Depth=2
	predpop	
	dstcr	0, r21
.LBB29_21:                              // %Flow20
                                        //   in Loop: Header=BB29_2 Depth=2
	djmpeqoff	0, r21, :.LBB29_29
// %bb.22:                              //   in Loop: Header=BB29_2 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB29_23
.LBB29_23:                              // %.preheader15
                                        //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB29_2 Depth=2
	djmpincsetup	0, 4, :.LBB29_25
	dstcr	0x200, pc.mode, south
.LBB29_25:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB29_2 Depth=2
	dcpc	r7, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	dstcr	0x200, pc.mode, south
	predpush	cr15, :.LBB29_28
// %bb.27:                              //   in Loop: Header=BB29_2 Depth=2
	cp	south, cr14
.LBB29_28:                              // %Flow19
                                        //   in Loop: Header=BB29_2 Depth=2
	predpop	
.LBB29_29:                              // %Flow21
                                        //   in Loop: Header=BB29_2 Depth=2
	dstcr	0, r21
.LBB29_30:                              // %Flow23
                                        //   in Loop: Header=BB29_2 Depth=2
	djmpeqoff	r21, 0, :.LBB29_38
// %bb.31:                              //   in Loop: Header=BB29_2 Depth=2
	djmpincsetup	0, 4, :.LBB29_32
	dstcr	0x200, pc.mode, south
.LBB29_32:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB29_2 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB29_34
.LBB29_34:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.35:                              //   in Loop: Header=BB29_2 Depth=2
	dcpc	r7, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB29_37
// %bb.36:                              //   in Loop: Header=BB29_2 Depth=2
	cp	south, cr14
.LBB29_37:                              // %Flow22
                                        //   in Loop: Header=BB29_2 Depth=2
	predpop	
.LBB29_38:                              //   in Loop: Header=BB29_2 Depth=2
	dshlb	r2, 2, r21
	daddi32	r2, 1, r2
	dcpc	r21, crp4
	addi32	crp2, crp4, crp4
	cp	cr14, [crp4]
.LBB29_39:                              //   in Loop: Header=BB29_2 Depth=2
	djmpincne	r30, 2, :.LBB29_2
// %bb.40:                              //   in Loop: Header=BB29_1 Depth=1
	dcmpneq32	r10, 7, r29
	dcsel	1, r6, r29
	dstcr	0x200, pc.mode, south
	dshlb	r29, 8, r8
	dstcr	0, r30
	daddi32	r8, r5, r18
	dstcr	0, r2
	dsubi32	r28, r18, r8
	daddi32	r18, r11, r9
	daddi32	r8, 15, r19
	dandb	r8, 12, r20
	dshrab	r19, 31, r21
	dcmpeq32	r20, 0, r20
	dshrlb	r21, 28, r20
	dsubi32	r13, r18, r21
	daddi32	r19, r20, r19
	dshrab	r21, 31, r20
	dandb	r19, -16, r19
	dshrlb	r20, 28, r20
	dcsel	r8, r19, r8
	dcmplt32	r28, r9, r28
	dshrab	r8, 31, r28
	daddi32	r21, r20, r9
	dshrlb	r28, 28, r28
	dshrab	r9, 4, r19
	daddi32	r8, r28, r8
	dshrlb	r18, 4, r28
	dshrab	r8, 4, r9
	dcp	[rp3], r8
	dstcr	0x9, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcsel	r9, 16, r9
	shlb	row, 4, cr13
	dcmplti32	r18, r14, r18
	stcr	0x1, bitwidthmode
	dsubi32	16, r9, r18
	dcsel	1, r19, r19
	addi32	cr13, col, cr13
	dstcr	0x0, pc.constant, south
	dstcr	0x800, pls.stride2, south
.LBB29_41:                              //   Parent Loop BB29_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB29_43 Depth 3
                                        //         Child Loop BB29_44 Depth 4
                                        //         Child Loop BB29_46 Depth 4
                                        //       Child Loop BB29_52 Depth 3
                                        //       Child Loop BB29_54 Depth 3
                                        //       Child Loop BB29_56 Depth 3
                                        //       Child Loop BB29_62 Depth 3
                                        //       Child Loop BB29_64 Depth 3
                                        //       Child Loop BB29_71 Depth 3
                                        //       Child Loop BB29_73 Depth 3
	djmpeqoff	0, r29, :.LBB29_48
// %bb.42:                              //   in Loop: Header=BB29_41 Depth=2
	dshlb	r2, 1, r21
	dstcr	0, r20
	dcpc	r21, crp4
	addi32	crp3, crp4, crp4
.LBB29_43:                              // %.preheader13
                                        //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB29_44 Depth 4
                                        //         Child Loop BB29_46 Depth 4
	dshlb	r20, 8, r21
	dshlb	r30, 12, r22
	daddi32	r21, r5, r21
	daddi32	r8, r22, r22
	dshrab	r21, 31, r23
	dsubi32	r13, r21, r24
	dshrlb	r23, 28, r23
	djmpincsetup	0, 16, :.LBB29_44
	daddi32	r21, r23, r23
	dcmplti32	r21, r14, r21
	dshlb	r23, 1, r21
	dshrab	r24, 31, r23
	dandb	r21, -32, r21
	dshrlb	r23, 28, r23
	daddi32	r22, r21, r21
	daddi32	r24, r23, r22
	dcp	r21, pls.addr, south
	dshrab	r22, 4, r22
	dcsel	16, r22, r21
	dcp	r21, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB29_44:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        //       Parent Loop BB29_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.45:                              //   in Loop: Header=BB29_43 Depth=3
	djmpincsetup	0, 4, :.LBB29_46
	dstcr	0x200, pc.mode, south
.LBB29_46:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        //       Parent Loop BB29_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.47:                              //   in Loop: Header=BB29_43 Depth=3
	cp	south.0z, [crp4.z+=1]
	daddi32	r2, 1, r2
	djmpincne	r20, r29, :.LBB29_43
.LBB29_48:                              // %.loopexit14
                                        //   in Loop: Header=BB29_41 Depth=2
	djmpneqoff	7, r10, :.LBB29_78
// %bb.49:                              //   in Loop: Header=BB29_41 Depth=2
	dshlb	r30, 12, r20
	dshlb	r28, 5, r21
	daddi32	r8, r20, r22
	dstcr	1, r20
	daddi32	r22, r21, r21
                                        // implicit-def: $cx14
	dcp	r21, pls.addr, south
	dcp	r19, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r9, 0, :.LBB29_69
// %bb.50:                              //   in Loop: Header=BB29_41 Depth=2
	dstcr	1, r20
                                        // implicit-def: $cx14
	djmpeqoff	0, r18, :.LBB29_60
// %bb.51:                              //   in Loop: Header=BB29_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB29_52
.LBB29_52:                              // %.preheader12
                                        //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.53:                              //   in Loop: Header=BB29_41 Depth=2
	djmpincsetup	0, 4, :.LBB29_54
	dstcr	0x200, pc.mode, south
.LBB29_54:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.55:                              //   in Loop: Header=BB29_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB29_56
.LBB29_56:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB29_41 Depth=2
	dcpc	r7, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB29_59
// %bb.58:                              //   in Loop: Header=BB29_41 Depth=2
	cp	south.0z, cr14
.LBB29_59:                              // %Flow9
                                        //   in Loop: Header=BB29_41 Depth=2
	predpop	
	dstcr	0, r20
.LBB29_60:                              // %Flow11
                                        //   in Loop: Header=BB29_41 Depth=2
	djmpeqoff	0, r20, :.LBB29_68
// %bb.61:                              //   in Loop: Header=BB29_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB29_62
.LBB29_62:                              // %.preheader11
                                        //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.63:                              //   in Loop: Header=BB29_41 Depth=2
	djmpincsetup	0, 4, :.LBB29_64
	dstcr	0x200, pc.mode, south
.LBB29_64:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB29_41 Depth=2
	dcpc	r7, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	dstcr	0x200, pc.mode, south
	predpush	cr15, :.LBB29_67
// %bb.66:                              //   in Loop: Header=BB29_41 Depth=2
	cp	south.0z, cr14
.LBB29_67:                              // %Flow10
                                        //   in Loop: Header=BB29_41 Depth=2
	predpop	
.LBB29_68:                              // %Flow12
                                        //   in Loop: Header=BB29_41 Depth=2
	dstcr	0, r20
.LBB29_69:                              // %Flow14
                                        //   in Loop: Header=BB29_41 Depth=2
	djmpeqoff	r20, 0, :.LBB29_77
// %bb.70:                              //   in Loop: Header=BB29_41 Depth=2
	djmpincsetup	0, 4, :.LBB29_71
	dstcr	0x200, pc.mode, south
.LBB29_71:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.72:                              //   in Loop: Header=BB29_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB29_73
.LBB29_73:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.74:                              //   in Loop: Header=BB29_41 Depth=2
	dcpc	r7, cr14
	cmplti32	cr13, cr14, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB29_76
// %bb.75:                              //   in Loop: Header=BB29_41 Depth=2
	cp	south.0z, cr14
.LBB29_76:                              // %Flow13
                                        //   in Loop: Header=BB29_41 Depth=2
	predpop	
.LBB29_77:                              //   in Loop: Header=BB29_41 Depth=2
	dshlb	r2, 1, r20
	daddi32	r2, 1, r2
	dcpc	r20, crp4
	addi32	crp3, crp4, crp4
	cp	cr14, [crp4.z]
.LBB29_78:                              //   in Loop: Header=BB29_41 Depth=2
	djmpincne	r30, 2, :.LBB29_41
// %bb.79:                              //   in Loop: Header=BB29_1 Depth=1
	cp	crp3, crp4
	cp	crp2, crp5
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 2, :.LBB29_80
	dstcr	0x200, pc.mode, south
.LBB29_80:                              //   Parent Loop BB29_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp	[crp5], cr13
	muli32lohi{16}	cr10, cr13, cr13
	sfs	cr13, cr11
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr13
	addi32	cr13, cr11, cr13
	stcr	0x1, bitwidthmode
	divi32{16}	cr11, cr13, cr13
	shlb	[crp4.s+=1], 10, cr14
	muli32lohi{10}	cr13, cr12, cr13
	stcr	0x2, bitwidthmode
	addi32.lb	cr14, cr13, [crp5+=1]
// %bb.81:                              //   in Loop: Header=BB29_1 Depth=1
	dcmplt32	r5, r16, r2
	dcsel	1, r6, r6
	dcp	[rp2], r7
	dshlb	r6, 8, r28
	shlb	row, 4, cr13
	daddi32	r28, r5, r29
	addi32	cr13, col, cr13
	dsubi32	r13, r29, r28
	daddi32	r29, r11, r30
	dshrab	r28, 31, r8
	dcmplt32	r30, r17, r30
	dshrlb	r8, 28, r30
	dshrab	r29, 31, r8
	daddi32	r28, r30, r28
	dshrlb	r8, 28, r30
	dshrab	r28, 4, r8
	daddi32	r29, r30, r30
	dcsel	1, r8, r28
	dcmplti32	r29, r14, r29
	dcsel	1, r8, r29
	dcmpneq32	r2, 0, r2
	dshrab	r30, 4, r30
	dcsel	0, 236, r2
	dstcr	0, r8
	dstcr	0, r9
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
.LBB29_82:                              //   Parent Loop BB29_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB29_84 Depth 3
                                        //         Child Loop BB29_85 Depth 4
                                        //         Child Loop BB29_87 Depth 4
                                        //       Child Loop BB29_103 Depth 3
                                        //       Child Loop BB29_105 Depth 3
                                        //       Child Loop BB29_95 Depth 3
	djmpeqoff	r6, 0, :.LBB29_89
// %bb.83:                              //   in Loop: Header=BB29_82 Depth=2
	dshlb	r9, 2, r19
	addi32	crp1, 24, crp4          //      
	dstcr	0, r18
	dcpc	r19, crp5
	addi32	crp4, crp5, crp4
.LBB29_84:                              // %.preheader
                                        //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_82 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB29_85 Depth 4
                                        //         Child Loop BB29_87 Depth 4
	dshlb	r18, 8, r19
	dmuli32	r8, r15, r20
	daddi32	r19, r5, r19
	djmpincsetup	0, 4, :.LBB29_85
	dshrab	r19, 31, r21
	dsubi32	r13, r19, r22
	dshrlb	r21, 28, r21
	daddi32	r7, r20, r20
	daddi32	r19, r21, r21
	dcmplti32	r19, r14, r19
	dshlb	r21, 2, r19
	dshrab	r22, 31, r21
	dandb	r19, -64, r19
	dshrlb	r21, 28, r21
	daddi32	r20, r19, r19
	daddi32	r22, r21, r20
	dcp	r19, pls.addr, north
	dshrab	r20, 4, r20
	dcsel	16, r20, r19
	dcp	r19, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	[crp4], north
	dstcr	0x200, pc.mode, north
.LBB29_85:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_82 Depth=2
                                        //       Parent Loop BB29_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.86:                              //   in Loop: Header=BB29_84 Depth=3
	djmpincsetup	0, 16, :.LBB29_87
	dstcr	0x300, pc.mode, north
.LBB29_87:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_82 Depth=2
                                        //       Parent Loop BB29_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.88:                              //   in Loop: Header=BB29_84 Depth=3
	addi32	crp4, 4, crp4
	daddi32	r9, 1, r9
	djmpincne	r18, r6, :.LBB29_84
.LBB29_89:                              // %.loopexit10
                                        //   in Loop: Header=BB29_82 Depth=2
	djmplt	r5, r16, :.LBB29_97
// %bb.90:                              //   in Loop: Header=BB29_82 Depth=2
	dmuli32	r8, r15, r18
	dshlb	r30, 6, r19
	dshlb	r9, 2, r20
	daddi32	r7, r18, r18
	addi32	crp1, 24, crp4          //      
	daddi32	r18, r19, r19
	dcpc	r20, crp5
	dcp	r19, pls.addr, north
	dcp	r29, pls.count1, north
	daddi32	r9, 1, r9
	addi32	crp4, crp5, crp4
	dstcr	1, r18
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r28, :.LBB29_91
// %bb.100:                             //   in Loop: Header=BB29_82 Depth=2
	dcpc	r2, cr14
	cmplti32	cr13, cr14, cr14
	predpush	cr14, :.LBB29_102
// %bb.101:                             //   in Loop: Header=BB29_82 Depth=2
	nrb	[crp4], north
.LBB29_102:                             //   in Loop: Header=BB29_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB29_103
	dstcr	0x200, pc.mode, north
.LBB29_103:                             //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.104:                             //   in Loop: Header=BB29_82 Depth=2
	dcp	r28, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB29_105
.LBB29_105:                             //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.106:                             // %Flow
                                        //   in Loop: Header=BB29_82 Depth=2
	dstcr	0, r18
.LBB29_91:                              // %Flow6
                                        //   in Loop: Header=BB29_82 Depth=2
	djmpeqoff	0, r18, :.LBB29_97
// %bb.92:                              //   in Loop: Header=BB29_82 Depth=2
	dcpc	r2, cr14
	cmplti32	cr13, cr14, cr14
	predpush	cr14, :.LBB29_94
// %bb.93:                              //   in Loop: Header=BB29_82 Depth=2
	nrb	[crp4], north
.LBB29_94:                              //   in Loop: Header=BB29_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB29_95
	dstcr	0x200, pc.mode, north
.LBB29_95:                              //   Parent Loop BB29_1 Depth=1
                                        //     Parent Loop BB29_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.96:                              //   in Loop: Header=BB29_82 Depth=2
	dstcr	0x300, pc.mode, north
.LBB29_97:                              // %.loopexit
                                        //   in Loop: Header=BB29_82 Depth=2
	djmpincne	r8, 2, :.LBB29_82
// %bb.98:                              //   in Loop: Header=BB29_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-438, r31
	djmpincne	r10, 8, r31
.LBB29_99:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r25
	dcp	[rp1 + 2], r24
	dcp	[rp1 + 4], r23
	dcp	[rp1 + 6], r22
	dcp	[rp2], r21
	daddi32	rp1, 40, rp2
	dcp	[rp2], r20
	daddi32	rp1, 48, rp2
	dcp	[rp2], r19
	daddi32	rp1, 56, rp2
	dcp	[rp2], r18
	daddi32	rp1, 64, rp2
	dcp	[rp2], r9
	daddi32	rp1, 72, rp2
	dcp	[rp2], r8
	daddi32	rp1, 80, rp1
	addi32	crp1, 72, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z25fused_exp_cast_multiply_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_
_Z25fused_exp_cast_multiply_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_: // @_Z25fused_exp_cast_multiply_1I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj2ELj1ELj2028EEES2_IS1_LS4_0EjLj64ELS5_1EJLj1ELj2ELj1ELj2028EEES6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -80, rp1
	daddi32	rp1, 72, rp2
	dstcr	0x2, mode
	addi32	crp1, -72, crp1         //     
	dcp	r11, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 64, rp2
	dcp	r10, rp4
	dcp	r9, [rp2]
	daddi32	rp1, 56, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 48, rp2
	dstcr	256, r11
	dcp	r19, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	2043, r13
	dcp	r20, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	1773, r14
	dcp	r21, [rp2]
	dcp	r12, rp2
	dstcr	2028, r12
	addi32	crp1, 24, crp2          //      
	dstcr	8128, r15
	cp	crp1, crp3
	stcr	65536, cr10
	dstcr	1772, r16
	dstcr	2029, r17
	dcp	r22, [rp1 + 6]
	dcp	r23, [rp1 + 4]
	dcp	r24, [rp1 + 2]
	dcp	r25, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x7f0, pls.stride2, north
.LBB30_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB30_2 Depth 2
                                        //       Child Loop BB30_4 Depth 3
                                        //         Child Loop BB30_5 Depth 4
                                        //         Child Loop BB30_7 Depth 4
                                        //       Child Loop BB30_13 Depth 3
                                        //       Child Loop BB30_15 Depth 3
                                        //       Child Loop BB30_17 Depth 3
                                        //       Child Loop BB30_23 Depth 3
                                        //       Child Loop BB30_25 Depth 3
                                        //       Child Loop BB30_32 Depth 3
                                        //       Child Loop BB30_34 Depth 3
                                        //     Child Loop BB30_41 Depth 2
                                        //       Child Loop BB30_43 Depth 3
                                        //         Child Loop BB30_44 Depth 4
                                        //         Child Loop BB30_46 Depth 4
                                        //       Child Loop BB30_52 Depth 3
                                        //       Child Loop BB30_54 Depth 3
                                        //       Child Loop BB30_56 Depth 3
                                        //       Child Loop BB30_62 Depth 3
                                        //       Child Loop BB30_64 Depth 3
                                        //       Child Loop BB30_71 Depth 3
                                        //       Child Loop BB30_73 Depth 3
                                        //     Child Loop BB30_80 Depth 2
                                        //     Child Loop BB30_82 Depth 2
                                        //       Child Loop BB30_84 Depth 3
                                        //         Child Loop BB30_85 Depth 4
                                        //         Child Loop BB30_87 Depth 4
                                        //       Child Loop BB30_103 Depth 3
                                        //       Child Loop BB30_105 Depth 3
                                        //       Child Loop BB30_95 Depth 3
	dcmplt32	6, r10, r5
	dshlb	r10, 8, r5
	dcsel	r12, r11, r28
	dsubi32	r12, r5, r6
	dcmpneq32	r10, 7, r7
	dshrlb	r6, 8, r6
	dcp	[rp4], r29
	dandb	r6, 255, r8
	dstcr	0x11, pls.mode, south
	dcsel	1, r8, r8
	dstcr	0x300, pc.mode, south
	dshlb	r8, 8, r9
	dstcr	0x200, pc.mode, north
	daddi32	r9, r5, r20
	shlb	row, 4, cr11
	dsubi32	r28, r20, r9
	daddi32	r20, r11, r18
	daddi32	r9, 15, r19
	dandb	r9, 12, r22
	dshrab	r19, 31, r21
	dstcr	0, r30
	dshrlb	r21, 28, r21
	dstcr	0, r2
	daddi32	r19, r21, r19
	dcmpeq32	r22, 0, r21
	dandb	r19, -16, r19
	dsubi32	r13, r20, r21
	dcsel	r9, r19, r9
	dcmplt32	r28, r18, r18
	dshrab	r9, 31, r18
	dshrab	r21, 31, r19
	dshrlb	r18, 28, r18
	dshrlb	r19, 28, r19
	daddi32	r9, r18, r9
	daddi32	r21, r19, r18
	dshrab	r9, 4, r9
	dshrab	r18, 4, r19
	dcsel	r9, 16, r9
	dcmplt32	r20, r14, r18
	dcsel	1, r19, r19
	dcmpneq32	r7, 0, r7
	addi32	cr11, col, cr11
	dsubi32	16, r9, r18
	dshrlb	r20, 4, r20
	dcsel	0, 236, r7
	stcr	0x2, bitwidthmode
	dstcr	0x0, pc.constant, south
	dstcr	0x7f0, pls.stride2, south
.LBB30_2:                               //   Parent Loop BB30_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB30_4 Depth 3
                                        //         Child Loop BB30_5 Depth 4
                                        //         Child Loop BB30_7 Depth 4
                                        //       Child Loop BB30_13 Depth 3
                                        //       Child Loop BB30_15 Depth 3
                                        //       Child Loop BB30_17 Depth 3
                                        //       Child Loop BB30_23 Depth 3
                                        //       Child Loop BB30_25 Depth 3
                                        //       Child Loop BB30_32 Depth 3
                                        //       Child Loop BB30_34 Depth 3
	djmpeqoff	0, r8, :.LBB30_9
// %bb.3:                               //   in Loop: Header=BB30_2 Depth=2
	dshlb	r2, 2, r22
	dstcr	0, r21
	dcpc	r22, crp4
	addi32	crp2, crp4, crp4
.LBB30_4:                               // %.preheader17
                                        //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB30_5 Depth 4
                                        //         Child Loop BB30_7 Depth 4
	dshlb	r21, 8, r23
	dmuli32	r30, r15, r22
	daddi32	r23, r5, r23
	djmpincsetup	0, 16, :.LBB30_5
	daddi32	r29, r22, r22
	dshlb	r23, 2, r24
	dsubi32	r13, r23, r25
	daddi32	r22, r24, r22
	dshrab	r25, 31, r24
	dcmplt32	r23, r14, r23
	dshrlb	r24, 28, r23
	dcp	r22, pls.addr, south
	daddi32	r25, r23, r22
	dshrab	r22, 4, r22
	dcsel	16, r22, r22
	dcp	r22, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
.LBB30_5:                               //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        //       Parent Loop BB30_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB30_4 Depth=3
	djmpincsetup	0, 4, :.LBB30_7
	dstcr	0x200, pc.mode, south
.LBB30_7:                               //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        //       Parent Loop BB30_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB30_4 Depth=3
	cp	south, [crp4+=1]
	daddi32	r2, 1, r2
	djmpincne	r21, r8, :.LBB30_4
.LBB30_9:                               // %.loopexit18
                                        //   in Loop: Header=BB30_2 Depth=2
	djmpneqoff	7, r10, :.LBB30_39
// %bb.10:                              //   in Loop: Header=BB30_2 Depth=2
	dmuli32	r30, r15, r22
	dshlb	r20, 6, r23
	dstcr	1, r21
	daddi32	r29, r22, r22
                                        // implicit-def: $cx12
	daddi32	r22, r23, r22
	dcp	r22, pls.addr, south
	dcp	r19, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r9, 0, :.LBB30_30
// %bb.11:                              //   in Loop: Header=BB30_2 Depth=2
	dstcr	1, r21
                                        // implicit-def: $cx12
	djmpeqoff	0, r18, :.LBB30_21
// %bb.12:                              //   in Loop: Header=BB30_2 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB30_13
.LBB30_13:                              // %.preheader16
                                        //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB30_2 Depth=2
	djmpincsetup	0, 4, :.LBB30_15
	dstcr	0x200, pc.mode, south
.LBB30_15:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB30_2 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB30_17
.LBB30_17:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB30_2 Depth=2
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB30_20
// %bb.19:                              //   in Loop: Header=BB30_2 Depth=2
	cp	south, cr12
.LBB30_20:                              // %Flow18
                                        //   in Loop: Header=BB30_2 Depth=2
	predpop	
	dstcr	0, r21
.LBB30_21:                              // %Flow20
                                        //   in Loop: Header=BB30_2 Depth=2
	djmpeqoff	0, r21, :.LBB30_29
// %bb.22:                              //   in Loop: Header=BB30_2 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB30_23
.LBB30_23:                              // %.preheader15
                                        //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB30_2 Depth=2
	djmpincsetup	0, 4, :.LBB30_25
	dstcr	0x200, pc.mode, south
.LBB30_25:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB30_2 Depth=2
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	dstcr	0x200, pc.mode, south
	predpush	cr13, :.LBB30_28
// %bb.27:                              //   in Loop: Header=BB30_2 Depth=2
	cp	south, cr12
.LBB30_28:                              // %Flow19
                                        //   in Loop: Header=BB30_2 Depth=2
	predpop	
.LBB30_29:                              // %Flow21
                                        //   in Loop: Header=BB30_2 Depth=2
	dstcr	0, r21
.LBB30_30:                              // %Flow23
                                        //   in Loop: Header=BB30_2 Depth=2
	djmpeqoff	r21, 0, :.LBB30_38
// %bb.31:                              //   in Loop: Header=BB30_2 Depth=2
	djmpincsetup	0, 4, :.LBB30_32
	dstcr	0x200, pc.mode, south
.LBB30_32:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB30_2 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB30_34
.LBB30_34:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.35:                              //   in Loop: Header=BB30_2 Depth=2
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB30_37
// %bb.36:                              //   in Loop: Header=BB30_2 Depth=2
	cp	south, cr12
.LBB30_37:                              // %Flow22
                                        //   in Loop: Header=BB30_2 Depth=2
	predpop	
.LBB30_38:                              //   in Loop: Header=BB30_2 Depth=2
	dshlb	r2, 2, r21
	daddi32	r2, 1, r2
	dcpc	r21, crp4
	addi32	crp2, crp4, crp4
	cp	cr12, [crp4]
.LBB30_39:                              //   in Loop: Header=BB30_2 Depth=2
	djmpincne	r30, 2, :.LBB30_2
// %bb.40:                              //   in Loop: Header=BB30_1 Depth=1
	dcmpneq32	r10, 7, r29
	dcsel	1, r6, r29
	dstcr	0x200, pc.mode, south
	dshlb	r29, 8, r8
	dstcr	0, r30
	daddi32	r8, r5, r18
	dstcr	0, r2
	dsubi32	r28, r18, r8
	daddi32	r18, r11, r9
	daddi32	r8, 15, r19
	dandb	r8, 12, r20
	dshrab	r19, 31, r21
	dcmpeq32	r20, 0, r20
	dshrlb	r21, 28, r20
	dsubi32	r13, r18, r21
	daddi32	r19, r20, r19
	dshrab	r21, 31, r20
	dandb	r19, -16, r19
	dshrlb	r20, 28, r20
	dcsel	r8, r19, r8
	dcmplt32	r28, r9, r28
	dshrab	r8, 31, r28
	daddi32	r21, r20, r9
	dshrlb	r28, 28, r28
	dshrab	r9, 4, r19
	daddi32	r8, r28, r8
	dshrlb	r18, 4, r28
	dshrab	r8, 4, r9
	dcp	[rp3], r8
	dstcr	0x9, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dcsel	r9, 16, r9
	shlb	row, 4, cr11
	dcmplti32	r18, r14, r18
	stcr	0x1, bitwidthmode
	dsubi32	16, r9, r18
	dcsel	1, r19, r19
	addi32	cr11, col, cr11
	dstcr	0x0, pc.constant, south
	dstcr	0x800, pls.stride2, south
.LBB30_41:                              //   Parent Loop BB30_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB30_43 Depth 3
                                        //         Child Loop BB30_44 Depth 4
                                        //         Child Loop BB30_46 Depth 4
                                        //       Child Loop BB30_52 Depth 3
                                        //       Child Loop BB30_54 Depth 3
                                        //       Child Loop BB30_56 Depth 3
                                        //       Child Loop BB30_62 Depth 3
                                        //       Child Loop BB30_64 Depth 3
                                        //       Child Loop BB30_71 Depth 3
                                        //       Child Loop BB30_73 Depth 3
	djmpeqoff	0, r29, :.LBB30_48
// %bb.42:                              //   in Loop: Header=BB30_41 Depth=2
	dshlb	r2, 1, r21
	dstcr	0, r20
	dcpc	r21, crp4
	addi32	crp3, crp4, crp4
.LBB30_43:                              // %.preheader13
                                        //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB30_44 Depth 4
                                        //         Child Loop BB30_46 Depth 4
	dshlb	r20, 8, r21
	dshlb	r30, 12, r22
	daddi32	r21, r5, r21
	daddi32	r8, r22, r22
	dshrab	r21, 31, r23
	dsubi32	r13, r21, r24
	dshrlb	r23, 28, r23
	djmpincsetup	0, 16, :.LBB30_44
	daddi32	r21, r23, r23
	dcmplti32	r21, r14, r21
	dshlb	r23, 1, r21
	dshrab	r24, 31, r23
	dandb	r21, -32, r21
	dshrlb	r23, 28, r23
	daddi32	r22, r21, r21
	daddi32	r24, r23, r22
	dcp	r21, pls.addr, south
	dshrab	r22, 4, r22
	dcsel	16, r22, r21
	dcp	r21, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB30_44:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        //       Parent Loop BB30_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.45:                              //   in Loop: Header=BB30_43 Depth=3
	djmpincsetup	0, 4, :.LBB30_46
	dstcr	0x200, pc.mode, south
.LBB30_46:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        //       Parent Loop BB30_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.47:                              //   in Loop: Header=BB30_43 Depth=3
	cp	south.0z, [crp4.z+=1]
	daddi32	r2, 1, r2
	djmpincne	r20, r29, :.LBB30_43
.LBB30_48:                              // %.loopexit14
                                        //   in Loop: Header=BB30_41 Depth=2
	djmpneqoff	7, r10, :.LBB30_78
// %bb.49:                              //   in Loop: Header=BB30_41 Depth=2
	dshlb	r30, 12, r20
	dshlb	r28, 5, r21
	daddi32	r8, r20, r22
	dstcr	1, r20
	daddi32	r22, r21, r21
                                        // implicit-def: $cx12
	dcp	r21, pls.addr, south
	dcp	r19, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r9, 0, :.LBB30_69
// %bb.50:                              //   in Loop: Header=BB30_41 Depth=2
	dstcr	1, r20
                                        // implicit-def: $cx12
	djmpeqoff	0, r18, :.LBB30_60
// %bb.51:                              //   in Loop: Header=BB30_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB30_52
.LBB30_52:                              // %.preheader12
                                        //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.53:                              //   in Loop: Header=BB30_41 Depth=2
	djmpincsetup	0, 4, :.LBB30_54
	dstcr	0x200, pc.mode, south
.LBB30_54:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.55:                              //   in Loop: Header=BB30_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB30_56
.LBB30_56:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB30_41 Depth=2
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB30_59
// %bb.58:                              //   in Loop: Header=BB30_41 Depth=2
	cp	south.0z, cr12
.LBB30_59:                              // %Flow9
                                        //   in Loop: Header=BB30_41 Depth=2
	predpop	
	dstcr	0, r20
.LBB30_60:                              // %Flow11
                                        //   in Loop: Header=BB30_41 Depth=2
	djmpeqoff	0, r20, :.LBB30_68
// %bb.61:                              //   in Loop: Header=BB30_41 Depth=2
	dcp	r9, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB30_62
.LBB30_62:                              // %.preheader11
                                        //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.63:                              //   in Loop: Header=BB30_41 Depth=2
	djmpincsetup	0, 4, :.LBB30_64
	dstcr	0x200, pc.mode, south
.LBB30_64:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB30_41 Depth=2
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	dstcr	0x200, pc.mode, south
	predpush	cr13, :.LBB30_67
// %bb.66:                              //   in Loop: Header=BB30_41 Depth=2
	cp	south.0z, cr12
.LBB30_67:                              // %Flow10
                                        //   in Loop: Header=BB30_41 Depth=2
	predpop	
.LBB30_68:                              // %Flow12
                                        //   in Loop: Header=BB30_41 Depth=2
	dstcr	0, r20
.LBB30_69:                              // %Flow14
                                        //   in Loop: Header=BB30_41 Depth=2
	djmpeqoff	r20, 0, :.LBB30_77
// %bb.70:                              //   in Loop: Header=BB30_41 Depth=2
	djmpincsetup	0, 4, :.LBB30_71
	dstcr	0x200, pc.mode, south
.LBB30_71:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.72:                              //   in Loop: Header=BB30_41 Depth=2
	dcp	r18, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB30_73
.LBB30_73:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.74:                              //   in Loop: Header=BB30_41 Depth=2
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB30_76
// %bb.75:                              //   in Loop: Header=BB30_41 Depth=2
	cp	south.0z, cr12
.LBB30_76:                              // %Flow13
                                        //   in Loop: Header=BB30_41 Depth=2
	predpop	
.LBB30_77:                              //   in Loop: Header=BB30_41 Depth=2
	dshlb	r2, 1, r20
	daddi32	r2, 1, r2
	dcpc	r20, crp4
	addi32	crp3, crp4, crp4
	cp	cr12, [crp4.z]
.LBB30_78:                              //   in Loop: Header=BB30_41 Depth=2
	djmpincne	r30, 2, :.LBB30_41
// %bb.79:                              //   in Loop: Header=BB30_1 Depth=1
	cp	crp3, crp4
	cp	crp2, crp5
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 2, :.LBB30_80
	dstcr	0x200, pc.mode, south
.LBB30_80:                              //   Parent Loop BB30_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sfs	[crp5], cr10
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr11
	stcr	0x1, bitwidthmode
	shlb	[crp4.s+=1], 10, cr12
	stcr	0x2, bitwidthmode
	muli32lohi{16}.lb	cr11, cr12, [crp5+=1]
// %bb.81:                              //   in Loop: Header=BB30_1 Depth=1
	dcmplt32	r5, r16, r2
	dcsel	1, r6, r6
	dcp	[rp2], r7
	dshlb	r6, 8, r28
	shlb	row, 4, cr11
	daddi32	r28, r5, r29
	addi32	cr11, col, cr11
	dsubi32	r13, r29, r28
	daddi32	r29, r11, r30
	dshrab	r28, 31, r8
	dcmplt32	r30, r17, r30
	dshrlb	r8, 28, r30
	dshrab	r29, 31, r8
	daddi32	r28, r30, r28
	dshrlb	r8, 28, r30
	dshrab	r28, 4, r8
	daddi32	r29, r30, r30
	dcsel	1, r8, r28
	dcmplti32	r29, r14, r29
	dcsel	1, r8, r29
	dcmpneq32	r2, 0, r2
	dshrab	r30, 4, r30
	dcsel	0, 236, r2
	dstcr	0, r8
	dstcr	0, r9
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
.LBB30_82:                              //   Parent Loop BB30_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB30_84 Depth 3
                                        //         Child Loop BB30_85 Depth 4
                                        //         Child Loop BB30_87 Depth 4
                                        //       Child Loop BB30_103 Depth 3
                                        //       Child Loop BB30_105 Depth 3
                                        //       Child Loop BB30_95 Depth 3
	djmpeqoff	r6, 0, :.LBB30_89
// %bb.83:                              //   in Loop: Header=BB30_82 Depth=2
	dshlb	r9, 2, r19
	addi32	crp1, 24, crp4          //      
	dstcr	0, r18
	dcpc	r19, crp5
	addi32	crp4, crp5, crp4
.LBB30_84:                              // %.preheader
                                        //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_82 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB30_85 Depth 4
                                        //         Child Loop BB30_87 Depth 4
	dshlb	r18, 8, r19
	dmuli32	r8, r15, r20
	daddi32	r19, r5, r19
	djmpincsetup	0, 4, :.LBB30_85
	dshrab	r19, 31, r21
	dsubi32	r13, r19, r22
	dshrlb	r21, 28, r21
	daddi32	r7, r20, r20
	daddi32	r19, r21, r21
	dcmplti32	r19, r14, r19
	dshlb	r21, 2, r19
	dshrab	r22, 31, r21
	dandb	r19, -64, r19
	dshrlb	r21, 28, r21
	daddi32	r20, r19, r19
	daddi32	r22, r21, r20
	dcp	r19, pls.addr, north
	dshrab	r20, 4, r20
	dcsel	16, r20, r19
	dcp	r19, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	[crp4], north
	dstcr	0x200, pc.mode, north
.LBB30_85:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_82 Depth=2
                                        //       Parent Loop BB30_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.86:                              //   in Loop: Header=BB30_84 Depth=3
	djmpincsetup	0, 16, :.LBB30_87
	dstcr	0x300, pc.mode, north
.LBB30_87:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_82 Depth=2
                                        //       Parent Loop BB30_84 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.88:                              //   in Loop: Header=BB30_84 Depth=3
	addi32	crp4, 4, crp4
	daddi32	r9, 1, r9
	djmpincne	r18, r6, :.LBB30_84
.LBB30_89:                              // %.loopexit10
                                        //   in Loop: Header=BB30_82 Depth=2
	djmplt	r5, r16, :.LBB30_97
// %bb.90:                              //   in Loop: Header=BB30_82 Depth=2
	dmuli32	r8, r15, r18
	dshlb	r30, 6, r19
	dshlb	r9, 2, r20
	daddi32	r7, r18, r18
	addi32	crp1, 24, crp4          //      
	daddi32	r18, r19, r19
	dcpc	r20, crp5
	dcp	r19, pls.addr, north
	dcp	r29, pls.count1, north
	daddi32	r9, 1, r9
	addi32	crp4, crp5, crp4
	dstcr	1, r18
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r28, :.LBB30_91
// %bb.100:                             //   in Loop: Header=BB30_82 Depth=2
	dcpc	r2, cr12
	cmplti32	cr11, cr12, cr12
	predpush	cr12, :.LBB30_102
// %bb.101:                             //   in Loop: Header=BB30_82 Depth=2
	nrb	[crp4], north
.LBB30_102:                             //   in Loop: Header=BB30_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB30_103
	dstcr	0x200, pc.mode, north
.LBB30_103:                             //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.104:                             //   in Loop: Header=BB30_82 Depth=2
	dcp	r28, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB30_105
.LBB30_105:                             //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.106:                             // %Flow
                                        //   in Loop: Header=BB30_82 Depth=2
	dstcr	0, r18
.LBB30_91:                              // %Flow6
                                        //   in Loop: Header=BB30_82 Depth=2
	djmpeqoff	0, r18, :.LBB30_97
// %bb.92:                              //   in Loop: Header=BB30_82 Depth=2
	dcpc	r2, cr12
	cmplti32	cr11, cr12, cr12
	predpush	cr12, :.LBB30_94
// %bb.93:                              //   in Loop: Header=BB30_82 Depth=2
	nrb	[crp4], north
.LBB30_94:                              //   in Loop: Header=BB30_82 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB30_95
	dstcr	0x200, pc.mode, north
.LBB30_95:                              //   Parent Loop BB30_1 Depth=1
                                        //     Parent Loop BB30_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.96:                              //   in Loop: Header=BB30_82 Depth=2
	dstcr	0x300, pc.mode, north
.LBB30_97:                              // %.loopexit
                                        //   in Loop: Header=BB30_82 Depth=2
	djmpincne	r8, 2, :.LBB30_82
// %bb.98:                              //   in Loop: Header=BB30_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-433, r31
	djmpincne	r10, 8, r31
.LBB30_99:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r25
	dcp	[rp1 + 2], r24
	dcp	[rp1 + 4], r23
	dcp	[rp1 + 6], r22
	dcp	[rp2], r21
	daddi32	rp1, 40, rp2
	dcp	[rp2], r20
	daddi32	rp1, 48, rp2
	dcp	[rp2], r19
	daddi32	rp1, 56, rp2
	dcp	[rp2], r18
	daddi32	rp1, 64, rp2
	dcp	[rp2], r9
	daddi32	rp1, 72, rp2
	dcp	[rp2], r8
	daddi32	rp1, 80, rp1
	addi32	crp1, 72, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z15fused_sigmoid_2I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2028EEES6_EvRT0_RT1_
_Z15fused_sigmoid_2I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2028EEES6_EvRT0_RT1_: // @_Z15fused_sigmoid_2I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2028EEES6_EvRT0_RT1_
// %bb.0:
	daddi32	rp1, -56, rp1
	daddi32	rp1, 48, rp2
	dstcr	0x2, mode
	addi32	crp1, -24, crp1         //     
	dcp	r10, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	0, r10
	dcp	r9, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	2028, r12
	dcp	r18, [rp2]
	dcp	r11, rp2
	dstcr	256, r11
	dstcr	2043, r13
	cp	crp1, crp2
	dstcr	1773, r14
	stcr	-65536, cr10
	stcr	65536, cr11
	dstcr	1772, r15
	dstcr	2029, r16
	dcp	r19, [rp1 + 6]
	dcp	r20, [rp1 + 4]
	dcp	r21, [rp1 + 2]
	dcp	r22, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x7f0, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x7f0, pls.stride2, north
.LBB31_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB31_54 Depth 2
                                        //       Child Loop BB31_55 Depth 3
                                        //       Child Loop BB31_52 Depth 3
                                        //     Child Loop BB31_6 Depth 2
                                        //     Child Loop BB31_8 Depth 2
                                        //     Child Loop BB31_10 Depth 2
                                        //     Child Loop BB31_16 Depth 2
                                        //     Child Loop BB31_18 Depth 2
                                        //     Child Loop BB31_25 Depth 2
                                        //     Child Loop BB31_27 Depth 2
                                        //     Child Loop BB31_34 Depth 2
                                        //       Child Loop BB31_35 Depth 3
                                        //       Child Loop BB31_37 Depth 3
                                        //     Child Loop BB31_59 Depth 2
                                        //     Child Loop BB31_61 Depth 2
                                        //     Child Loop BB31_47 Depth 2
	dcmplt32	6, r10, r17
	dshlb	r10, 8, r17
	dcsel	r12, r11, r28
	dcmpeq32	r10, 7, r5
	dsubi32	r12, r17, r5
	dcp	[rp3], r29
	dshrlb	r5, 8, r5
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dandb	r5, 255, r6
	shlb	row, 4, cr12
	dcsel	236, 0, r7
	cp	crp2, crp3
	dstcr	0, r30
	dcsel	r6, 1, r6
	addi32	cr12, col, cr12
	dstcr	0x0, pc.constant, south
	djmpeqoff	0, r6, :.LBB31_2
.LBB31_54:                              // %.preheader8
                                        //   Parent Loop BB31_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB31_55 Depth 3
                                        //       Child Loop BB31_52 Depth 3
	dshlb	r30, 8, r2
	djmpincsetup	0, 16, :.LBB31_55
	daddi32	r2, r17, r2
	dsubi32	r13, r2, r9
	dshlb	r2, 2, r8
	dcmplt32	r2, r14, r2
	dshrab	r9, 31, r2
	daddi32	r29, r8, r8
	dshrlb	r2, 28, r2
	dcp	r8, pls.addr, south
	daddi32	r9, r2, r2
	dshrab	r2, 4, r2
	dcsel	16, r2, r2
	dcp	r2, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB31_55:                              //   Parent Loop BB31_1 Depth=1
                                        //     Parent Loop BB31_54 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.51:                              //   in Loop: Header=BB31_54 Depth=2
	djmpincsetup	0, 4, :.LBB31_52
	dstcr	0x200, pc.mode, south
.LBB31_52:                              //   Parent Loop BB31_1 Depth=1
                                        //     Parent Loop BB31_54 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.53:                              //   in Loop: Header=BB31_54 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r30, r6, :.LBB31_54
.LBB31_2:                               // %.loopexit9
                                        //   in Loop: Header=BB31_1 Depth=1
	djmpneqoff	7, r10, :.LBB31_32
// %bb.3:                               //   in Loop: Header=BB31_1 Depth=1
	dshlb	r6, 8, r2
	dstcr	1, r30
	daddi32	r2, r17, r2
                                        // implicit-def: $cx13
	dsubi32	r28, r2, r8
	dshrlb	r2, 4, r22
	daddi32	r8, 15, r18
	dsubi32	r13, r2, r21
	dshrab	r18, 31, r20
	daddi32	r2, r11, r9
	dshrlb	r20, 28, r20
	dcmplt32	r2, r14, r2
	daddi32	r18, r20, r18
	dshlb	r22, 6, r20
	dshrab	r21, 31, r2
	daddi32	r29, r20, r29
	dandb	r8, 12, r19
	dcp	r29, pls.addr, south
	dshrlb	r2, 28, r29
	dandb	r18, -16, r18
	daddi32	r21, r29, r29
	dshrab	r29, 4, r29
	dcsel	1, r29, r29
	dcmpeq32	r19, 0, r2
	dcp	r29, pls.count1, south
	dcsel	r8, r18, r29
	dcmplt32	r28, r9, r28
	dshrab	r29, 31, r28
	dcp	[rp3 + 1], dependencyid
	dshrlb	r28, 28, r28
	dstcr	0x100, plsstatus, south
	daddi32	r29, r28, r28
	dcp	flowid, [rp3 + 1]
	dshrab	r28, 4, r28
	dcsel	r28, 16, r29
	dsubi32	16, r29, r28
	djmpeqoff	r29, 0, :.LBB31_23
// %bb.4:                               //   in Loop: Header=BB31_1 Depth=1
	dstcr	1, r30
                                        // implicit-def: $cx13
	dstcr	0x300, pc.mode, south
	djmpeqoff	0, r28, :.LBB31_14
// %bb.5:                               //   in Loop: Header=BB31_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB31_6
.LBB31_6:                               // %.preheader7
                                        //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB31_1 Depth=1
	djmpincsetup	0, 4, :.LBB31_8
	dstcr	0x200, pc.mode, south
.LBB31_8:                               //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB31_1 Depth=1
	dcp	r28, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB31_10
.LBB31_10:                              //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB31_1 Depth=1
	dcpc	r7, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	predpush	cr14, :.LBB31_13
// %bb.12:                              //   in Loop: Header=BB31_1 Depth=1
	cp	south, cr13
.LBB31_13:                              // %Flow8
                                        //   in Loop: Header=BB31_1 Depth=1
	predpop	
	dstcr	0, r30
.LBB31_14:                              // %Flow10
                                        //   in Loop: Header=BB31_1 Depth=1
	djmpeqoff	0, r30, :.LBB31_22
// %bb.15:                              //   in Loop: Header=BB31_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB31_16
.LBB31_16:                              // %.preheader6
                                        //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB31_1 Depth=1
	djmpincsetup	0, 4, :.LBB31_18
	dstcr	0x200, pc.mode, south
.LBB31_18:                              //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB31_1 Depth=1
	dcpc	r7, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	dstcr	0x200, pc.mode, south
	predpush	cr14, :.LBB31_21
// %bb.20:                              //   in Loop: Header=BB31_1 Depth=1
	cp	south, cr13
.LBB31_21:                              // %Flow9
                                        //   in Loop: Header=BB31_1 Depth=1
	predpop	
.LBB31_22:                              // %Flow11
                                        //   in Loop: Header=BB31_1 Depth=1
	dstcr	0, r30
.LBB31_23:                              // %Flow13
                                        //   in Loop: Header=BB31_1 Depth=1
	djmpeqoff	r30, 0, :.LBB31_31
// %bb.24:                              //   in Loop: Header=BB31_1 Depth=1
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, 4, :.LBB31_25
	dstcr	0x200, pc.mode, south
.LBB31_25:                              //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB31_1 Depth=1
	dcp	r28, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB31_27
.LBB31_27:                              //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB31_1 Depth=1
	dcpc	r7, cr13
	cmplti32	cr12, cr13, cr12
	stcr	0, cr13
	predpush	cr12, :.LBB31_30
// %bb.29:                              //   in Loop: Header=BB31_1 Depth=1
	cp	south, cr13
.LBB31_30:                              // %Flow12
                                        //   in Loop: Header=BB31_1 Depth=1
	predpop	
.LBB31_31:                              //   in Loop: Header=BB31_1 Depth=1
	dshlb	r6, 2, r6
	dcpc	r6, crp3
	addi32	crp2, crp3, crp3
	cp	cr13, [crp3]
.LBB31_32:                              //   in Loop: Header=BB31_1 Depth=1
	dstcr	0x200, pc.mode, south
	cp	[crp1], cr12
	dcmplt32	r17, r15, r6
	muli32lohi{16}	cr10, cr12, cr12
	sfs	cr12, cr11
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr12
	addi32	cr12, cr11, cr13
	dcsel	1, r5, r6
	divi32{16}	cr11, cr13, [crp1]
	dcsel	0, 236, r5
	dcp	[rp2], r7
	shlb	row, 4, cr12
	addi32	cr12, col, cr12
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	djmpeqoff	r6, 0, :.LBB31_41
// %bb.33:                              //   in Loop: Header=BB31_1 Depth=1
	divi32{16}	cr11, cr13, cr13
	dstcr	0, r28
	orb	crp2, 4, crp3
.LBB31_34:                              // %.preheader
                                        //   Parent Loop BB31_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB31_35 Depth 3
                                        //       Child Loop BB31_37 Depth 3
	dshlb	r28, 8, r29
	djmpincsetup	0, 4, :.LBB31_35
	daddi32	r29, r17, r29
	dshrab	r29, 31, r30
	dsubi32	r13, r29, r2
	dshrlb	r30, 28, r30
	dshrab	r2, 31, r8
	daddi32	r29, r30, r30
	dshrlb	r8, 28, r8
	dshlb	r30, 2, r30
	daddi32	r2, r8, r2
	dandb	r30, -64, r30
	dshrab	r2, 4, r2
	dcmplti32	r29, r14, r29
	daddi32	r7, r30, r30
	dcsel	16, r2, r29
	dcp	r30, pls.addr, north
	dcp	r29, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	cr13, north
	dstcr	0x200, pc.mode, north
.LBB31_35:                              //   Parent Loop BB31_1 Depth=1
                                        //     Parent Loop BB31_34 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.36:                              //   in Loop: Header=BB31_34 Depth=2
	djmpincsetup	0, 16, :.LBB31_37
	dstcr	0x300, pc.mode, north
.LBB31_37:                              //   Parent Loop BB31_1 Depth=1
                                        //     Parent Loop BB31_34 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.38:                              //   in Loop: Header=BB31_34 Depth=2
	daddi32	r28, 1, r28
	dstcr	1, r29
                                        // implicit-def: $cx13
	djmpeqoff	r28, r6, :.LBB31_40
// %bb.39:                              //   in Loop: Header=BB31_34 Depth=2
	cp	[crp3+=1], cr13
	dstcr	0, r29
.LBB31_40:                              // %Flow6
                                        //   in Loop: Header=BB31_34 Depth=2
	djmpeqoff	r29, 0, :.LBB31_34
.LBB31_41:                              // %.loopexit5
                                        //   in Loop: Header=BB31_1 Depth=1
	djmplt	r17, r15, :.LBB31_49
// %bb.42:                              //   in Loop: Header=BB31_1 Depth=1
	dshlb	r6, 8, r28
	dshlb	r6, 2, r29
	daddi32	r28, r17, r17
	dstcr	1, r6
	dshrab	r17, 31, r28
	dcpc	r29, crp3
	dshrlb	r28, 28, r28
	dsubi32	r13, r17, r29
	daddi32	r17, r28, r28
	daddi32	r17, r11, r30
	dcmplti32	r17, r14, r17
	dshrab	r28, 4, r17
	dshrab	r29, 31, r28
	dshlb	r17, 6, r17
	dshrlb	r28, 28, r28
	daddi32	r7, r17, r7
	daddi32	r29, r28, r17
	dcp	r7, pls.addr, north
	dshrab	r17, 4, r17
	addi32	crp2, crp3, crp3
	dcsel	1, r17, r28
	dcmplt32	r30, r16, r29
	dcp	r28, pls.count1, north
	dcsel	1, r17, r17
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r17, :.LBB31_43
// %bb.56:                              //   in Loop: Header=BB31_1 Depth=1
	dcpc	r5, cr13
	cmplti32	cr12, cr13, cr13
	predpush	cr13, :.LBB31_58
// %bb.57:                              //   in Loop: Header=BB31_1 Depth=1
	nrb	[crp3], north
.LBB31_58:                              //   in Loop: Header=BB31_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB31_59
	dstcr	0x200, pc.mode, north
.LBB31_59:                              //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.60:                              //   in Loop: Header=BB31_1 Depth=1
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB31_61
.LBB31_61:                              //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.62:                              // %Flow
                                        //   in Loop: Header=BB31_1 Depth=1
	dstcr	0, r6
.LBB31_43:                              // %Flow4
                                        //   in Loop: Header=BB31_1 Depth=1
	djmpeqoff	0, r6, :.LBB31_49
// %bb.44:                              //   in Loop: Header=BB31_1 Depth=1
	dcpc	r5, cr13
	cmplti32	cr12, cr13, cr12
	predpush	cr12, :.LBB31_46
// %bb.45:                              //   in Loop: Header=BB31_1 Depth=1
	nrb	[crp3], north
.LBB31_46:                              //   in Loop: Header=BB31_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB31_47
	dstcr	0x200, pc.mode, north
.LBB31_47:                              //   Parent Loop BB31_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.48:                              //   in Loop: Header=BB31_1 Depth=1
	dstcr	0x300, pc.mode, north
.LBB31_49:                              // %.loopexit
                                        //   in Loop: Header=BB31_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-263, r31
	djmpincne	r10, 8, r31
.LBB31_50:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r22
	dcp	[rp1 + 2], r21
	dcp	[rp1 + 4], r20
	dcp	[rp1 + 6], r19
	dcp	[rp2], r18
	daddi32	rp1, 40, rp2
	dcp	[rp2], r9
	daddi32	rp1, 48, rp2
	dcp	[rp2], r8
	daddi32	rp1, 56, rp1
	addi32	crp1, 24, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z15fused_sigmoid_3I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj2028EEES6_EvRT0_RT1_
_Z15fused_sigmoid_3I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj2028EEES6_EvRT0_RT1_: // @_Z15fused_sigmoid_3I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj80ELj1ELj2028EEES6_EvRT0_RT1_
// %bb.0:
	daddi32	rp1, -88, rp1
	daddi32	rp1, 80, rp2
	dstcr	0x2, mode
	stcr	-336, cr10
	dcp	r10, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 72, rp2
	addi32	crp1, cr10, crp1        //     
	dcp	r9, [rp2]
	daddi32	rp1, 64, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 56, rp2
	dstcr	2028, r12
	dcp	r19, [rp2]
	daddi32	rp1, 48, rp2
	dstcr	2043, r13
	dcp	r20, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	1773, r14
	dcp	r21, [rp2]
	daddi32	rp1, 32, rp2
	addi32	crp1, 16, crp2          //      
	dcp	r22, [rp2]
	dcp	r11, rp2
	dstcr	256, r11
	dstcr	8128, r15
	stcr	-65536, cr10
	stcr	65536, cr11
	dstcr	1772, r16
	dstcr	2029, r17
	dcp	r23, [rp1 + 6]
	dcp	r24, [rp1 + 4]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x7f0, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x7f0, pls.stride2, north
.LBB32_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB32_2 Depth 2
                                        //       Child Loop BB32_4 Depth 3
                                        //         Child Loop BB32_5 Depth 4
                                        //         Child Loop BB32_7 Depth 4
                                        //       Child Loop BB32_13 Depth 3
                                        //       Child Loop BB32_15 Depth 3
                                        //       Child Loop BB32_17 Depth 3
                                        //       Child Loop BB32_23 Depth 3
                                        //       Child Loop BB32_25 Depth 3
                                        //       Child Loop BB32_32 Depth 3
                                        //       Child Loop BB32_34 Depth 3
                                        //     Child Loop BB32_41 Depth 2
                                        //     Child Loop BB32_43 Depth 2
                                        //       Child Loop BB32_45 Depth 3
                                        //         Child Loop BB32_46 Depth 4
                                        //         Child Loop BB32_48 Depth 4
                                        //       Child Loop BB32_64 Depth 3
                                        //       Child Loop BB32_66 Depth 3
                                        //       Child Loop BB32_56 Depth 3
	dcmplt32	6, r10, r5
	dshlb	r10, 8, r5
	dcsel	r12, r11, r2
	dsubi32	r12, r5, r6
	dcmpeq32	r10, 7, r19
	dshrlb	r6, 8, r6
	dcp	[rp3], r7
	dandb	r6, 255, r30
	dstcr	0x11, pls.mode, south
	dcsel	r30, 1, r30
	dstcr	0x300, pc.mode, south
	dshlb	r30, 8, r8
	dstcr	0x200, pc.mode, north
	daddi32	r8, r5, r9
	shlb	row, 4, cr12
	dsubi32	r2, r9, r8
	daddi32	r9, r11, r18
	daddi32	r8, 15, r20
	dandb	r8, 12, r22
	dshrab	r20, 31, r21
	dcmpeq32	r22, 0, r22
	dshrlb	r21, 28, r21
	dstcr	0, r28
	daddi32	r20, r21, r20
	dsubi32	r13, r9, r21
	dandb	r20, -16, r20
	dshrab	r21, 31, r22
	dcsel	r8, r20, r8
	dcmplt32	r2, r18, r2
	dshrab	r8, 31, r2
	dshrlb	r22, 28, r18
	dshrlb	r2, 28, r2
	daddi32	r21, r18, r18
	daddi32	r8, r2, r2
	dshrab	r18, 4, r18
	dshrab	r2, 4, r8
	dshrlb	r9, 4, r2
	dcsel	r8, 16, r8
	dcmplt32	r9, r14, r9
	dcsel	1, r18, r18
	dcmpneq32	r19, 0, r19
	dstcr	0, r29
	addi32	cr12, col, cr12
	dsubi32	16, r8, r9
	dcsel	236, 0, r19
	dstcr	0x0, pc.constant, south
.LBB32_2:                               //   Parent Loop BB32_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB32_4 Depth 3
                                        //         Child Loop BB32_5 Depth 4
                                        //         Child Loop BB32_7 Depth 4
                                        //       Child Loop BB32_13 Depth 3
                                        //       Child Loop BB32_15 Depth 3
                                        //       Child Loop BB32_17 Depth 3
                                        //       Child Loop BB32_23 Depth 3
                                        //       Child Loop BB32_25 Depth 3
                                        //       Child Loop BB32_32 Depth 3
                                        //       Child Loop BB32_34 Depth 3
	djmpeqoff	0, r30, :.LBB32_9
// %bb.3:                               //   in Loop: Header=BB32_2 Depth=2
	dshlb	r29, 2, r21
	dstcr	0, r20
	dcpc	r21, crp3
	addi32	crp2, crp3, crp3
.LBB32_4:                               // %.preheader10
                                        //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB32_5 Depth 4
                                        //         Child Loop BB32_7 Depth 4
	dshlb	r20, 8, r22
	dmuli32	r28, r15, r21
	daddi32	r22, r5, r22
	djmpincsetup	0, 16, :.LBB32_5
	daddi32	r7, r21, r21
	dshlb	r22, 2, r23
	dsubi32	r13, r22, r24
	daddi32	r21, r23, r21
	dshrab	r24, 31, r23
	dcmplt32	r22, r14, r22
	dshrlb	r23, 28, r22
	dcp	r21, pls.addr, south
	daddi32	r24, r22, r21
	dshrab	r21, 4, r21
	dcsel	16, r21, r21
	dcp	r21, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB32_5:                               //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        //       Parent Loop BB32_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB32_4 Depth=3
	djmpincsetup	0, 4, :.LBB32_7
	dstcr	0x200, pc.mode, south
.LBB32_7:                               //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        //       Parent Loop BB32_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB32_4 Depth=3
	cp	south, [crp3+=1]
	daddi32	r29, 1, r29
	djmpincne	r20, r30, :.LBB32_4
.LBB32_9:                               // %.loopexit11
                                        //   in Loop: Header=BB32_2 Depth=2
	djmpneqoff	7, r10, :.LBB32_39
// %bb.10:                              //   in Loop: Header=BB32_2 Depth=2
	dmuli32	r28, r15, r21
	dshlb	r2, 6, r22
	dstcr	1, r20
	daddi32	r7, r21, r21
                                        // implicit-def: $cx13
	daddi32	r21, r22, r21
	dcp	r21, pls.addr, south
	dcp	r18, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r8, 0, :.LBB32_30
// %bb.11:                              //   in Loop: Header=BB32_2 Depth=2
	dstcr	1, r20
                                        // implicit-def: $cx13
	djmpeqoff	0, r9, :.LBB32_21
// %bb.12:                              //   in Loop: Header=BB32_2 Depth=2
	dcp	r8, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB32_13
.LBB32_13:                              // %.preheader9
                                        //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB32_2 Depth=2
	djmpincsetup	0, 4, :.LBB32_15
	dstcr	0x200, pc.mode, south
.LBB32_15:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB32_2 Depth=2
	dcp	r9, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB32_17
.LBB32_17:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB32_2 Depth=2
	dcpc	r19, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	predpush	cr14, :.LBB32_20
// %bb.19:                              //   in Loop: Header=BB32_2 Depth=2
	cp	south, cr13
.LBB32_20:                              // %Flow7
                                        //   in Loop: Header=BB32_2 Depth=2
	predpop	
	dstcr	0, r20
.LBB32_21:                              // %Flow9
                                        //   in Loop: Header=BB32_2 Depth=2
	djmpeqoff	0, r20, :.LBB32_29
// %bb.22:                              //   in Loop: Header=BB32_2 Depth=2
	dcp	r8, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB32_23
.LBB32_23:                              // %.preheader8
                                        //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB32_2 Depth=2
	djmpincsetup	0, 4, :.LBB32_25
	dstcr	0x200, pc.mode, south
.LBB32_25:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB32_2 Depth=2
	dcpc	r19, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	dstcr	0x200, pc.mode, south
	predpush	cr14, :.LBB32_28
// %bb.27:                              //   in Loop: Header=BB32_2 Depth=2
	cp	south, cr13
.LBB32_28:                              // %Flow8
                                        //   in Loop: Header=BB32_2 Depth=2
	predpop	
.LBB32_29:                              // %Flow10
                                        //   in Loop: Header=BB32_2 Depth=2
	dstcr	0, r20
.LBB32_30:                              // %Flow12
                                        //   in Loop: Header=BB32_2 Depth=2
	djmpeqoff	r20, 0, :.LBB32_38
// %bb.31:                              //   in Loop: Header=BB32_2 Depth=2
	djmpincsetup	0, 4, :.LBB32_32
	dstcr	0x200, pc.mode, south
.LBB32_32:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB32_2 Depth=2
	dcp	r9, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB32_34
.LBB32_34:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.35:                              //   in Loop: Header=BB32_2 Depth=2
	dcpc	r19, cr13
	cmplti32	cr12, cr13, cr14
	stcr	0, cr13
	predpush	cr14, :.LBB32_37
// %bb.36:                              //   in Loop: Header=BB32_2 Depth=2
	cp	south, cr13
.LBB32_37:                              // %Flow11
                                        //   in Loop: Header=BB32_2 Depth=2
	predpop	
.LBB32_38:                              //   in Loop: Header=BB32_2 Depth=2
	dshlb	r29, 2, r20
	daddi32	r29, 1, r29
	dcpc	r20, crp3
	addi32	crp2, crp3, crp3
	cp	cr13, [crp3]
.LBB32_39:                              //   in Loop: Header=BB32_2 Depth=2
	djmpincne	r28, 80, :.LBB32_2
// %bb.40:                              //   in Loop: Header=BB32_1 Depth=1
	cp	crp2, crp3
	djmpincsetup	0, 80, :.LBB32_41
	dstcr	0x200, pc.mode, south
.LBB32_41:                              //   Parent Loop BB32_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp	[crp3], cr12
	muli32lohi{16}	cr10, cr12, cr12
	sfs	cr12, cr11
	expnscl	726817, 16
	expint	363408, 8
	expint	181704, 4
	expint	90852, 2
	expint	45426, 1
	expfrc	26573, 1
	expfrc	14624, 2
	expfrc	7719, 3
	expfrc	3973, 4
	expfrc	2017, 5
	expfrc	1016, 6
	expfrc	510, 7
	expfrc	256, 8
	expfrc	128, 9
	expfrc	64, 10
	expfrc	32, 11
	expfrc	16, 12
	expfrc	8, 13
	expfrc	4, 14
	expfrc	2, 15
	expfrc	1, 16
	sfe	sfy, cr12
	addi32	cr12, cr11, cr12
	divi32{16}.lb	cr11, cr12, [crp3+=1]
// %bb.42:                              //   in Loop: Header=BB32_1 Depth=1
	dcmplt32	r5, r16, r2
	dcsel	1, r6, r6
	dcp	[rp2], r7
	dshlb	r6, 8, r28
	shlb	row, 4, cr12
	daddi32	r28, r5, r29
	addi32	cr12, col, cr12
	dsubi32	r13, r29, r28
	daddi32	r29, r11, r30
	dshrab	r28, 31, r8
	dcmplt32	r30, r17, r30
	dshrlb	r8, 28, r30
	dshrab	r29, 31, r8
	daddi32	r28, r30, r28
	dshrlb	r8, 28, r30
	dshrab	r28, 4, r8
	daddi32	r29, r30, r30
	dcsel	1, r8, r28
	dcmplti32	r29, r14, r29
	dcsel	1, r8, r29
	dcmpneq32	r2, 0, r2
	dshrab	r30, 4, r30
	dcsel	0, 236, r2
	dstcr	0, r8
	dstcr	0, r9
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
.LBB32_43:                              //   Parent Loop BB32_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB32_45 Depth 3
                                        //         Child Loop BB32_46 Depth 4
                                        //         Child Loop BB32_48 Depth 4
                                        //       Child Loop BB32_64 Depth 3
                                        //       Child Loop BB32_66 Depth 3
                                        //       Child Loop BB32_56 Depth 3
	djmpeqoff	r6, 0, :.LBB32_50
// %bb.44:                              //   in Loop: Header=BB32_43 Depth=2
	dshlb	r9, 2, r19
	dstcr	0, r18
	dcpc	r19, crp3
	addi32	crp2, crp3, crp3
.LBB32_45:                              // %.preheader
                                        //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_43 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB32_46 Depth 4
                                        //         Child Loop BB32_48 Depth 4
	dshlb	r18, 8, r19
	dmuli32	r8, r15, r20
	daddi32	r19, r5, r19
	djmpincsetup	0, 4, :.LBB32_46
	dshrab	r19, 31, r21
	dsubi32	r13, r19, r22
	dshrlb	r21, 28, r21
	daddi32	r7, r20, r20
	daddi32	r19, r21, r21
	dcmplti32	r19, r14, r19
	dshlb	r21, 2, r19
	dshrab	r22, 31, r21
	dandb	r19, -64, r19
	dshrlb	r21, 28, r21
	daddi32	r20, r19, r19
	daddi32	r22, r21, r20
	dcp	r19, pls.addr, north
	dshrab	r20, 4, r20
	dcsel	16, r20, r19
	dcp	r19, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	[crp3], north
	dstcr	0x200, pc.mode, north
.LBB32_46:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_43 Depth=2
                                        //       Parent Loop BB32_45 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.47:                              //   in Loop: Header=BB32_45 Depth=3
	djmpincsetup	0, 16, :.LBB32_48
	dstcr	0x300, pc.mode, north
.LBB32_48:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_43 Depth=2
                                        //       Parent Loop BB32_45 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.49:                              //   in Loop: Header=BB32_45 Depth=3
	addi32	crp3, 4, crp3
	daddi32	r9, 1, r9
	djmpincne	r18, r6, :.LBB32_45
.LBB32_50:                              // %.loopexit7
                                        //   in Loop: Header=BB32_43 Depth=2
	djmplt	r5, r16, :.LBB32_58
// %bb.51:                              //   in Loop: Header=BB32_43 Depth=2
	dmuli32	r8, r15, r18
	dshlb	r30, 6, r19
	dshlb	r9, 2, r20
	daddi32	r7, r18, r18
	addi32	crp1, 16, crp3          //      
	daddi32	r18, r19, r19
	dcpc	r20, crp4
	dcp	r19, pls.addr, north
	dcp	r29, pls.count1, north
	daddi32	r9, 1, r9
	addi32	crp3, crp4, crp3
	dstcr	1, r18
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r28, :.LBB32_52
// %bb.61:                              //   in Loop: Header=BB32_43 Depth=2
	dcpc	r2, cr13
	cmplti32	cr12, cr13, cr13
	predpush	cr13, :.LBB32_63
// %bb.62:                              //   in Loop: Header=BB32_43 Depth=2
	nrb	[crp3], north
.LBB32_63:                              //   in Loop: Header=BB32_43 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB32_64
	dstcr	0x200, pc.mode, north
.LBB32_64:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_43 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB32_43 Depth=2
	dcp	r28, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB32_66
.LBB32_66:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_43 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.67:                              // %Flow
                                        //   in Loop: Header=BB32_43 Depth=2
	dstcr	0, r18
.LBB32_52:                              // %Flow4
                                        //   in Loop: Header=BB32_43 Depth=2
	djmpeqoff	0, r18, :.LBB32_58
// %bb.53:                              //   in Loop: Header=BB32_43 Depth=2
	dcpc	r2, cr13
	cmplti32	cr12, cr13, cr13
	predpush	cr13, :.LBB32_55
// %bb.54:                              //   in Loop: Header=BB32_43 Depth=2
	nrb	[crp3], north
.LBB32_55:                              //   in Loop: Header=BB32_43 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB32_56
	dstcr	0x200, pc.mode, north
.LBB32_56:                              //   Parent Loop BB32_1 Depth=1
                                        //     Parent Loop BB32_43 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.57:                              //   in Loop: Header=BB32_43 Depth=2
	dstcr	0x300, pc.mode, north
.LBB32_58:                              // %.loopexit
                                        //   in Loop: Header=BB32_43 Depth=2
	djmpincne	r8, 80, :.LBB32_43
// %bb.59:                              //   in Loop: Header=BB32_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-284, r31
	djmpincne	r10, 8, r31
.LBB32_60:
	daddi32	rp1, 32, rp2
	dcp	[rp1 + 4], r24
	dcp	[rp1 + 6], r23
	dcp	[rp2], r22
	daddi32	rp1, 40, rp2
	dcp	[rp2], r21
	daddi32	rp1, 48, rp2
	dcp	[rp2], r20
	daddi32	rp1, 56, rp2
	dcp	[rp2], r19
	daddi32	rp1, 64, rp2
	dcp	[rp2], r18
	daddi32	rp1, 72, rp2
	dcp	[rp2], r9
	daddi32	rp1, 80, rp2
	dcp	[rp2], r8
	daddi32	rp1, 88, rp1
	stcr	336, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z31fused_fixed_point_multiply_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_EvRT0_RT1_
_Z31fused_fixed_point_multiply_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_EvRT0_RT1_: // @_Z31fused_fixed_point_multiply_castI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_EvRT0_RT1_
// %bb.0:
	daddi32	rp1, -56, rp1
	daddi32	rp1, 48, rp2
	dstcr	0x2, mode
	addi32	crp1, -24, crp1         //     
	dcp	r10, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	0, r10
	dcp	r9, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	2535, r12
	dcp	r18, [rp2]
	dcp	r11, rp2
	dstcr	256, r11
	dstcr	2550, r13
	cp	crp1, crp2
	dstcr	2280, r14
	stcr	16384, cr10
	dstcr	2279, r15
	dstcr	2536, r16
	dcp	r19, [rp1 + 6]
	dcp	r20, [rp1 + 4]
	dcp	r21, [rp1 + 2]
	dcp	r22, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x9f0, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x9f0, pls.stride2, north
.LBB33_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB33_54 Depth 2
                                        //       Child Loop BB33_55 Depth 3
                                        //       Child Loop BB33_52 Depth 3
                                        //     Child Loop BB33_6 Depth 2
                                        //     Child Loop BB33_8 Depth 2
                                        //     Child Loop BB33_10 Depth 2
                                        //     Child Loop BB33_16 Depth 2
                                        //     Child Loop BB33_18 Depth 2
                                        //     Child Loop BB33_25 Depth 2
                                        //     Child Loop BB33_27 Depth 2
                                        //     Child Loop BB33_34 Depth 2
                                        //       Child Loop BB33_35 Depth 3
                                        //       Child Loop BB33_37 Depth 3
                                        //     Child Loop BB33_59 Depth 2
                                        //     Child Loop BB33_61 Depth 2
                                        //     Child Loop BB33_47 Depth 2
	dcmplt32	8, r10, r17
	dshlb	r10, 8, r17
	dcsel	r12, r11, r28
	dcmpeq32	r10, 9, r5
	dsubi32	r12, r17, r5
	dcp	[rp3], r29
	dshrlb	r5, 8, r5
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dandb	r5, 255, r6
	shlb	row, 4, cr11
	dcsel	231, 0, r7
	cp	crp2, crp3
	dstcr	0, r30
	dcsel	r6, 1, r6
	addi32	cr11, col, cr11
	dstcr	0x0, pc.constant, south
	djmpeqoff	0, r6, :.LBB33_2
.LBB33_54:                              // %.preheader8
                                        //   Parent Loop BB33_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB33_55 Depth 3
                                        //       Child Loop BB33_52 Depth 3
	dshlb	r30, 8, r2
	djmpincsetup	0, 16, :.LBB33_55
	daddi32	r2, r17, r2
	dsubi32	r13, r2, r9
	dshlb	r2, 2, r8
	dcmplt32	r2, r14, r2
	dshrab	r9, 31, r2
	daddi32	r29, r8, r8
	dshrlb	r2, 28, r2
	dcp	r8, pls.addr, south
	daddi32	r9, r2, r2
	dshrab	r2, 4, r2
	dcsel	16, r2, r2
	dcp	r2, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB33_55:                              //   Parent Loop BB33_1 Depth=1
                                        //     Parent Loop BB33_54 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.51:                              //   in Loop: Header=BB33_54 Depth=2
	djmpincsetup	0, 4, :.LBB33_52
	dstcr	0x200, pc.mode, south
.LBB33_52:                              //   Parent Loop BB33_1 Depth=1
                                        //     Parent Loop BB33_54 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.53:                              //   in Loop: Header=BB33_54 Depth=2
	cp	south, [crp3+=1]
	djmpincne	r30, r6, :.LBB33_54
.LBB33_2:                               // %.loopexit9
                                        //   in Loop: Header=BB33_1 Depth=1
	djmpneqoff	9, r10, :.LBB33_32
// %bb.3:                               //   in Loop: Header=BB33_1 Depth=1
	dshlb	r6, 8, r2
	dstcr	1, r30
	daddi32	r2, r17, r2
                                        // implicit-def: $cx12
	dsubi32	r28, r2, r8
	dshrlb	r2, 4, r22
	daddi32	r8, 15, r18
	dsubi32	r13, r2, r21
	dshrab	r18, 31, r20
	daddi32	r2, r11, r9
	dshrlb	r20, 28, r20
	dcmplt32	r2, r14, r2
	daddi32	r18, r20, r18
	dshlb	r22, 6, r20
	dshrab	r21, 31, r2
	daddi32	r29, r20, r29
	dandb	r8, 7, r19
	dcp	r29, pls.addr, south
	dshrlb	r2, 28, r29
	dandb	r18, -16, r18
	daddi32	r21, r29, r29
	dshrab	r29, 4, r29
	dcsel	1, r29, r29
	dcmpeq32	r19, 0, r2
	dcp	r29, pls.count1, south
	dcsel	r8, r18, r29
	dcmplt32	r28, r9, r28
	dshrab	r29, 31, r28
	dcp	[rp3 + 1], dependencyid
	dshrlb	r28, 28, r28
	dstcr	0x100, plsstatus, south
	daddi32	r29, r28, r28
	dcp	flowid, [rp3 + 1]
	dshrab	r28, 4, r28
	dcsel	r28, 16, r29
	dsubi32	16, r29, r28
	djmpeqoff	r29, 0, :.LBB33_23
// %bb.4:                               //   in Loop: Header=BB33_1 Depth=1
	dstcr	1, r30
                                        // implicit-def: $cx12
	dstcr	0x300, pc.mode, south
	djmpeqoff	0, r28, :.LBB33_14
// %bb.5:                               //   in Loop: Header=BB33_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB33_6
.LBB33_6:                               // %.preheader7
                                        //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.7:                               //   in Loop: Header=BB33_1 Depth=1
	djmpincsetup	0, 4, :.LBB33_8
	dstcr	0x200, pc.mode, south
.LBB33_8:                               //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.9:                               //   in Loop: Header=BB33_1 Depth=1
	dcp	r28, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB33_10
.LBB33_10:                              //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.11:                              //   in Loop: Header=BB33_1 Depth=1
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB33_13
// %bb.12:                              //   in Loop: Header=BB33_1 Depth=1
	cp	south, cr12
.LBB33_13:                              // %Flow8
                                        //   in Loop: Header=BB33_1 Depth=1
	predpop	
	dstcr	0, r30
.LBB33_14:                              // %Flow10
                                        //   in Loop: Header=BB33_1 Depth=1
	djmpeqoff	0, r30, :.LBB33_22
// %bb.15:                              //   in Loop: Header=BB33_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB33_16
.LBB33_16:                              // %.preheader6
                                        //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB33_1 Depth=1
	djmpincsetup	0, 4, :.LBB33_18
	dstcr	0x200, pc.mode, south
.LBB33_18:                              //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.19:                              //   in Loop: Header=BB33_1 Depth=1
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr13
	stcr	0, cr12
	dstcr	0x200, pc.mode, south
	predpush	cr13, :.LBB33_21
// %bb.20:                              //   in Loop: Header=BB33_1 Depth=1
	cp	south, cr12
.LBB33_21:                              // %Flow9
                                        //   in Loop: Header=BB33_1 Depth=1
	predpop	
.LBB33_22:                              // %Flow11
                                        //   in Loop: Header=BB33_1 Depth=1
	dstcr	0, r30
.LBB33_23:                              // %Flow13
                                        //   in Loop: Header=BB33_1 Depth=1
	djmpeqoff	r30, 0, :.LBB33_31
// %bb.24:                              //   in Loop: Header=BB33_1 Depth=1
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, 4, :.LBB33_25
	dstcr	0x200, pc.mode, south
.LBB33_25:                              //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB33_1 Depth=1
	dcp	r28, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB33_27
.LBB33_27:                              //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.28:                              //   in Loop: Header=BB33_1 Depth=1
	dcpc	r7, cr12
	cmplti32	cr11, cr12, cr11
	stcr	0, cr12
	predpush	cr11, :.LBB33_30
// %bb.29:                              //   in Loop: Header=BB33_1 Depth=1
	cp	south, cr12
.LBB33_30:                              // %Flow12
                                        //   in Loop: Header=BB33_1 Depth=1
	predpop	
.LBB33_31:                              //   in Loop: Header=BB33_1 Depth=1
	dshlb	r6, 2, r6
	dcpc	r6, crp3
	addi32	crp2, crp3, crp3
	cp	cr12, [crp3]
.LBB33_32:                              //   in Loop: Header=BB33_1 Depth=1
	dstcr	0x200, pc.mode, south
	cp	[crp1], cr12
	dcmplt32	r17, r15, r6
	muli32lohi{15}	cr12, cr10, [crp1]
	dcsel	1, r5, r7
	dcp	[rp2], r6
	shlb	row, 4, cr11
	dcsel	0, 231, r5
	addi32	cr11, col, cr11
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	djmpeqoff	r7, 0, :.LBB33_41
// %bb.33:                              //   in Loop: Header=BB33_1 Depth=1
	muli32lohi{15}	cr12, cr10, cr12
	orb	crp2, 4, crp3
	dstcr	0, r28
.LBB33_34:                              // %.preheader
                                        //   Parent Loop BB33_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB33_35 Depth 3
                                        //       Child Loop BB33_37 Depth 3
	dshlb	r28, 8, r29
	djmpincsetup	0, 4, :.LBB33_35
	daddi32	r29, r17, r29
	dshrab	r29, 31, r30
	dsubi32	r13, r29, r2
	dshrlb	r30, 28, r30
	dshrab	r2, 31, r8
	daddi32	r29, r30, r30
	dshrlb	r8, 28, r8
	dshlb	r30, 2, r30
	daddi32	r2, r8, r2
	dandb	r30, -64, r30
	dshrab	r2, 4, r2
	dcmplti32	r29, r14, r29
	daddi32	r6, r30, r30
	dcsel	16, r2, r29
	dcp	r30, pls.addr, north
	dcp	r29, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	cr12, north
	dstcr	0x200, pc.mode, north
.LBB33_35:                              //   Parent Loop BB33_1 Depth=1
                                        //     Parent Loop BB33_34 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.36:                              //   in Loop: Header=BB33_34 Depth=2
	djmpincsetup	0, 16, :.LBB33_37
	dstcr	0x300, pc.mode, north
.LBB33_37:                              //   Parent Loop BB33_1 Depth=1
                                        //     Parent Loop BB33_34 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.38:                              //   in Loop: Header=BB33_34 Depth=2
	daddi32	r28, 1, r28
	dstcr	1, r29
                                        // implicit-def: $cx12
	djmpeqoff	r28, r7, :.LBB33_40
// %bb.39:                              //   in Loop: Header=BB33_34 Depth=2
	cp	[crp3+=1], cr12
	dstcr	0, r29
.LBB33_40:                              // %Flow6
                                        //   in Loop: Header=BB33_34 Depth=2
	djmpeqoff	r29, 0, :.LBB33_34
.LBB33_41:                              // %.loopexit5
                                        //   in Loop: Header=BB33_1 Depth=1
	djmplt	r17, r15, :.LBB33_49
// %bb.42:                              //   in Loop: Header=BB33_1 Depth=1
	dshlb	r7, 8, r28
	dshlb	r7, 2, r29
	daddi32	r28, r17, r17
	dstcr	1, r7
	dshrab	r17, 31, r28
	dcpc	r29, crp3
	dshrlb	r28, 28, r28
	dsubi32	r13, r17, r29
	daddi32	r17, r28, r28
	daddi32	r17, r11, r30
	dcmplti32	r17, r14, r17
	dshrab	r28, 4, r17
	dshrab	r29, 31, r28
	dshlb	r17, 6, r17
	dshrlb	r28, 28, r28
	daddi32	r6, r17, r6
	daddi32	r29, r28, r17
	dcp	r6, pls.addr, north
	dshrab	r17, 4, r17
	addi32	crp2, crp3, crp3
	dcsel	1, r17, r28
	dcmplt32	r30, r16, r29
	dcp	r28, pls.count1, north
	dcsel	1, r17, r17
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r17, :.LBB33_43
// %bb.56:                              //   in Loop: Header=BB33_1 Depth=1
	dcpc	r5, cr12
	cmplti32	cr11, cr12, cr12
	predpush	cr12, :.LBB33_58
// %bb.57:                              //   in Loop: Header=BB33_1 Depth=1
	nrb	[crp3], north
.LBB33_58:                              //   in Loop: Header=BB33_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB33_59
	dstcr	0x200, pc.mode, north
.LBB33_59:                              //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.60:                              //   in Loop: Header=BB33_1 Depth=1
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB33_61
.LBB33_61:                              //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.62:                              // %Flow
                                        //   in Loop: Header=BB33_1 Depth=1
	dstcr	0, r7
.LBB33_43:                              // %Flow4
                                        //   in Loop: Header=BB33_1 Depth=1
	djmpeqoff	0, r7, :.LBB33_49
// %bb.44:                              //   in Loop: Header=BB33_1 Depth=1
	dcpc	r5, cr12
	cmplti32	cr11, cr12, cr11
	predpush	cr11, :.LBB33_46
// %bb.45:                              //   in Loop: Header=BB33_1 Depth=1
	nrb	[crp3], north
.LBB33_46:                              //   in Loop: Header=BB33_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB33_47
	dstcr	0x200, pc.mode, north
.LBB33_47:                              //   Parent Loop BB33_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.48:                              //   in Loop: Header=BB33_1 Depth=1
	dstcr	0x300, pc.mode, north
.LBB33_49:                              // %.loopexit
                                        //   in Loop: Header=BB33_1 Depth=1
	dstcr	0x200, pc.mode, north
	djmpincne	r10, 10, :.LBB33_1
// %bb.50:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r22
	dcp	[rp1 + 2], r21
	dcp	[rp1 + 4], r20
	dcp	[rp1 + 6], r19
	dcp	[rp2], r18
	daddi32	rp1, 40, rp2
	dcp	[rp2], r9
	daddi32	rp1, 48, rp2
	dcp	[rp2], r8
	daddi32	rp1, 56, rp1
	addi32	crp1, 24, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z14fused_subtractI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
_Z14fused_subtractI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_: // @_Z14fused_subtractI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -56, rp1
	daddi32	rp1, 48, rp2
	dstcr	0x2, mode
	addi32	crp1, -48, crp1         //     
	dcp	r11, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 40, rp2
	dcp	r10, rp4
	dcp	r9, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	dcp	r12, rp2
	dstcr	256, r11
	dstcr	2535, r12
	dstcr	2550, r13
	addi32	crp1, 24, crp2          //      
	dstcr	2280, r14
	cp	crp1, crp3
	dstcr	2279, r15
	dstcr	2536, r16
	dcp	r19, [rp1 + 6]
	dcp	r20, [rp1 + 4]
	dcp	r21, [rp1 + 2]
	dcp	r22, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x9f0, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x9f0, pls.stride2, north
.LBB34_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB34_7 Depth 2
                                        //       Child Loop BB34_8 Depth 3
                                        //       Child Loop BB34_5 Depth 3
                                        //     Child Loop BB34_12 Depth 2
                                        //     Child Loop BB34_14 Depth 2
                                        //     Child Loop BB34_16 Depth 2
                                        //     Child Loop BB34_22 Depth 2
                                        //     Child Loop BB34_24 Depth 2
                                        //     Child Loop BB34_31 Depth 2
                                        //     Child Loop BB34_33 Depth 2
                                        //     Child Loop BB34_91 Depth 2
                                        //       Child Loop BB34_92 Depth 3
                                        //       Child Loop BB34_89 Depth 3
                                        //     Child Loop BB34_43 Depth 2
                                        //     Child Loop BB34_45 Depth 2
                                        //     Child Loop BB34_47 Depth 2
                                        //     Child Loop BB34_53 Depth 2
                                        //     Child Loop BB34_55 Depth 2
                                        //     Child Loop BB34_62 Depth 2
                                        //     Child Loop BB34_64 Depth 2
                                        //     Child Loop BB34_71 Depth 2
                                        //       Child Loop BB34_72 Depth 3
                                        //       Child Loop BB34_74 Depth 3
                                        //     Child Loop BB34_96 Depth 2
                                        //     Child Loop BB34_98 Depth 2
                                        //     Child Loop BB34_84 Depth 2
	dcmplt32	8, r10, r17
	dshlb	r10, 8, r17
	dcsel	r12, r11, r6
	dcmpneq32	r10, 9, r5
	dsubi32	r12, r17, r5
	dcp	[rp4], r29
	dshrlb	r5, 8, r5
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dandb	r5, 255, r7
	shlb	row, 4, cr10
	dcsel	0, 231, r28
	cp	crp2, crp4
	dstcr	0, r30
	dcsel	1, r7, r7
	addi32	cr10, col, cr10
	dstcr	0x0, pc.constant, south
	djmpeqoff	0, r7, :.LBB34_2
.LBB34_7:                               // %.preheader14
                                        //   Parent Loop BB34_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB34_8 Depth 3
                                        //       Child Loop BB34_5 Depth 3
	dshlb	r30, 8, r2
	djmpincsetup	0, 16, :.LBB34_8
	daddi32	r2, r17, r2
	dsubi32	r13, r2, r9
	dshlb	r2, 2, r8
	dcmplt32	r2, r14, r2
	dshrab	r9, 31, r2
	daddi32	r29, r8, r8
	dshrlb	r2, 28, r2
	dcp	r8, pls.addr, south
	daddi32	r9, r2, r2
	dshrab	r2, 4, r2
	dcsel	16, r2, r2
	dcp	r2, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
.LBB34_8:                               //   Parent Loop BB34_1 Depth=1
                                        //     Parent Loop BB34_7 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.4:                               //   in Loop: Header=BB34_7 Depth=2
	djmpincsetup	0, 4, :.LBB34_5
	dstcr	0x200, pc.mode, south
.LBB34_5:                               //   Parent Loop BB34_1 Depth=1
                                        //     Parent Loop BB34_7 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB34_7 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r30, r7, :.LBB34_7
.LBB34_2:                               // %.loopexit15
                                        //   in Loop: Header=BB34_1 Depth=1
	djmpneqoff	9, r10, :.LBB34_3
// %bb.9:                               //   in Loop: Header=BB34_1 Depth=1
	dshlb	r7, 8, r30
	dstcr	1, r2
	daddi32	r30, r17, r30
                                        // implicit-def: $cx11
	dsubi32	r6, r30, r8
	dshrlb	r30, 4, r22
	daddi32	r8, 15, r18
	dsubi32	r13, r30, r21
	dshrab	r18, 31, r20
	daddi32	r30, r11, r9
	dshrlb	r20, 28, r20
	dcmplt32	r30, r14, r30
	daddi32	r18, r20, r18
	dshlb	r22, 6, r20
	dshrab	r21, 31, r30
	daddi32	r29, r20, r29
	dandb	r8, 7, r19
	dcp	r29, pls.addr, south
	dshrlb	r30, 28, r29
	dandb	r18, -16, r18
	daddi32	r21, r29, r29
	dshrab	r29, 4, r29
	dcsel	1, r29, r29
	dcmpeq32	r19, 0, r30
	dcp	r29, pls.count1, south
	dcsel	r8, r18, r29
	dcmplt32	r6, r9, r30
	dshrab	r29, 31, r30
	dcp	[rp4 + 1], dependencyid
	dshrlb	r30, 28, r30
	dstcr	0x100, plsstatus, south
	daddi32	r29, r30, r29
	dcp	flowid, [rp4 + 1]
	dshrab	r29, 4, r29
	dcsel	r29, 16, r30
	dsubi32	16, r30, r29
	djmpeqoff	r30, 0, :.LBB34_29
// %bb.10:                              //   in Loop: Header=BB34_1 Depth=1
	dstcr	1, r2
                                        // implicit-def: $cx11
	dstcr	0x300, pc.mode, south
	djmpeqoff	0, r29, :.LBB34_20
// %bb.11:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r30, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB34_12
.LBB34_12:                              // %.preheader13
                                        //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.13:                              //   in Loop: Header=BB34_1 Depth=1
	djmpincsetup	0, 4, :.LBB34_14
	dstcr	0x200, pc.mode, south
.LBB34_14:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.15:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r29, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB34_16
.LBB34_16:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	predpush	cr12, :.LBB34_19
// %bb.18:                              //   in Loop: Header=BB34_1 Depth=1
	cp	south, cr11
.LBB34_19:                              // %Flow19
                                        //   in Loop: Header=BB34_1 Depth=1
	predpop	
	dstcr	0, r2
.LBB34_20:                              // %Flow21
                                        //   in Loop: Header=BB34_1 Depth=1
	djmpeqoff	0, r2, :.LBB34_28
// %bb.21:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r30, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB34_22
.LBB34_22:                              // %.preheader12
                                        //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.23:                              //   in Loop: Header=BB34_1 Depth=1
	djmpincsetup	0, 4, :.LBB34_24
	dstcr	0x200, pc.mode, south
.LBB34_24:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.25:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	dstcr	0x200, pc.mode, south
	predpush	cr12, :.LBB34_27
// %bb.26:                              //   in Loop: Header=BB34_1 Depth=1
	cp	south, cr11
.LBB34_27:                              // %Flow20
                                        //   in Loop: Header=BB34_1 Depth=1
	predpop	
.LBB34_28:                              // %Flow22
                                        //   in Loop: Header=BB34_1 Depth=1
	dstcr	0, r2
.LBB34_29:                              // %Flow24
                                        //   in Loop: Header=BB34_1 Depth=1
	djmpeqoff	r2, 0, :.LBB34_37
// %bb.30:                              //   in Loop: Header=BB34_1 Depth=1
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, 4, :.LBB34_31
	dstcr	0x200, pc.mode, south
.LBB34_31:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.32:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r29, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB34_33
.LBB34_33:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr10
	stcr	0, cr11
	predpush	cr10, :.LBB34_36
// %bb.35:                              //   in Loop: Header=BB34_1 Depth=1
	cp	south, cr11
.LBB34_36:                              // %Flow23
                                        //   in Loop: Header=BB34_1 Depth=1
	predpop	
.LBB34_37:                              //   in Loop: Header=BB34_1 Depth=1
	dshlb	r7, 2, r29
	dstcr	231, r28
	dcp	r5, r7
	dcpc	r29, crp4
	addi32	crp2, crp4, crp4
	cp	cr11, [crp4]
	djmp	:.LBB34_38
.LBB34_3:                               //   in Loop: Header=BB34_1 Depth=1
	dstcr	0, r28
	dstcr	1, r7
.LBB34_38:                              //   in Loop: Header=BB34_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r29
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr10
	dstcr	0, r30
	cp	crp3, crp4
	addi32	cr10, col, cr10
	dstcr	0x0, pc.constant, south
	djmpeqoff	0, r7, :.LBB34_39
.LBB34_91:                              // %.preheader10
                                        //   Parent Loop BB34_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB34_92 Depth 3
                                        //       Child Loop BB34_89 Depth 3
	dshlb	r30, 8, r2
	djmpincsetup	0, 16, :.LBB34_92
	daddi32	r2, r17, r2
	dsubi32	r13, r2, r9
	dshlb	r2, 2, r8
	dcmplt32	r2, r14, r2
	dshrab	r9, 31, r2
	daddi32	r29, r8, r8
	dshrlb	r2, 28, r2
	dcp	r8, pls.addr, south
	daddi32	r9, r2, r2
	dshrab	r2, 4, r2
	dcsel	16, r2, r2
	dcp	r2, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB34_92:                              //   Parent Loop BB34_1 Depth=1
                                        //     Parent Loop BB34_91 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.88:                              //   in Loop: Header=BB34_91 Depth=2
	djmpincsetup	0, 4, :.LBB34_89
	dstcr	0x200, pc.mode, south
.LBB34_89:                              //   Parent Loop BB34_1 Depth=1
                                        //     Parent Loop BB34_91 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.90:                              //   in Loop: Header=BB34_91 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r30, r7, :.LBB34_91
.LBB34_39:                              // %.loopexit11
                                        //   in Loop: Header=BB34_1 Depth=1
	djmpeqoff	0, r28, :.LBB34_69
// %bb.40:                              //   in Loop: Header=BB34_1 Depth=1
	dshlb	r7, 8, r2
	dstcr	1, r30
	daddi32	r2, r17, r2
                                        // implicit-def: $cx11
	dshrlb	r2, 4, r21
	dsubi32	r6, r2, r8
	dsubi32	r13, r2, r19
	dshlb	r21, 6, r21
	daddi32	r2, r11, r9
	dcmplti32	r2, r14, r2
	daddi32	r8, 15, r18
	daddi32	r29, r21, r29
	dshrab	r19, 31, r2
	dshrab	r18, 31, r20
	dcp	r29, pls.addr, south
	dshrlb	r2, 28, r29
	dshrlb	r20, 28, r20
	daddi32	r19, r29, r29
	daddi32	r18, r20, r18
	dshrab	r29, 4, r29
	dandb	r8, 7, r20
	dandb	r18, -16, r18
	dcsel	1, r29, r29
	dcmpeq32	r20, 0, r2
	dcp	r29, pls.count1, south
	dcsel	r8, r18, r29
	dcmplt32	r6, r9, r6
	dshrab	r29, 31, r6
	dcp	[rp3 + 1], dependencyid
	dshrlb	r6, 28, r6
	dstcr	0x100, plsstatus, south
	daddi32	r29, r6, r6
	dcp	flowid, [rp3 + 1]
	dshrab	r6, 4, r6
	dcsel	r6, 16, r29
	dsubi32	16, r29, r6
	djmpeqoff	r29, 0, :.LBB34_60
// %bb.41:                              //   in Loop: Header=BB34_1 Depth=1
	dstcr	1, r30
                                        // implicit-def: $cx11
	dstcr	0x300, pc.mode, south
	djmpeqoff	0, r6, :.LBB34_51
// %bb.42:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB34_43
.LBB34_43:                              // %.preheader9
                                        //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB34_1 Depth=1
	djmpincsetup	0, 4, :.LBB34_45
	dstcr	0x200, pc.mode, south
.LBB34_45:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.46:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB34_47
.LBB34_47:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.48:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	predpush	cr12, :.LBB34_50
// %bb.49:                              //   in Loop: Header=BB34_1 Depth=1
	cp	south, cr11
.LBB34_50:                              // %Flow10
                                        //   in Loop: Header=BB34_1 Depth=1
	predpop	
	dstcr	0, r30
.LBB34_51:                              // %Flow12
                                        //   in Loop: Header=BB34_1 Depth=1
	djmpeqoff	0, r30, :.LBB34_59
// %bb.52:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB34_53
.LBB34_53:                              // %.preheader8
                                        //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.54:                              //   in Loop: Header=BB34_1 Depth=1
	djmpincsetup	0, 4, :.LBB34_55
	dstcr	0x200, pc.mode, south
.LBB34_55:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.56:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	dstcr	0x200, pc.mode, south
	predpush	cr12, :.LBB34_58
// %bb.57:                              //   in Loop: Header=BB34_1 Depth=1
	cp	south, cr11
.LBB34_58:                              // %Flow11
                                        //   in Loop: Header=BB34_1 Depth=1
	predpop	
.LBB34_59:                              // %Flow13
                                        //   in Loop: Header=BB34_1 Depth=1
	dstcr	0, r30
.LBB34_60:                              // %Flow15
                                        //   in Loop: Header=BB34_1 Depth=1
	djmpeqoff	r30, 0, :.LBB34_68
// %bb.61:                              //   in Loop: Header=BB34_1 Depth=1
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, 4, :.LBB34_62
	dstcr	0x200, pc.mode, south
.LBB34_62:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.63:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB34_64
.LBB34_64:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr10
	stcr	0, cr11
	predpush	cr10, :.LBB34_67
// %bb.66:                              //   in Loop: Header=BB34_1 Depth=1
	cp	south, cr11
.LBB34_67:                              // %Flow14
                                        //   in Loop: Header=BB34_1 Depth=1
	predpop	
.LBB34_68:                              //   in Loop: Header=BB34_1 Depth=1
	dshlb	r7, 2, r6
	dcpc	r6, crp4
	addi32	crp3, crp4, crp4
	cp	cr11, [crp4]
.LBB34_69:                              //   in Loop: Header=BB34_1 Depth=1
	dstcr	0x200, pc.mode, south
	cp	[crp1 + 6], cr11
	cp	[crp1], cr12
	dcmplt32	r17, r15, r6
	subi32	cr11, cr12, [crp1 + 6]
	dcsel	1, r5, r7
	dcp	[rp2], r6
	shlb	row, 4, cr10
	dcsel	0, 231, r5
	addi32	cr10, col, cr10
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	djmpeqoff	r7, 0, :.LBB34_78
// %bb.70:                              //   in Loop: Header=BB34_1 Depth=1
	subi32	cr11, cr12, cr11
	orb	crp2, 4, crp4
	dstcr	0, r28
.LBB34_71:                              // %.preheader
                                        //   Parent Loop BB34_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB34_72 Depth 3
                                        //       Child Loop BB34_74 Depth 3
	dshlb	r28, 8, r29
	djmpincsetup	0, 4, :.LBB34_72
	daddi32	r29, r17, r29
	dshrab	r29, 31, r30
	dsubi32	r13, r29, r2
	dshrlb	r30, 28, r30
	dshrab	r2, 31, r8
	daddi32	r29, r30, r30
	dshrlb	r8, 28, r8
	dshlb	r30, 2, r30
	daddi32	r2, r8, r2
	dandb	r30, -64, r30
	dshrab	r2, 4, r2
	dcmplti32	r29, r14, r29
	daddi32	r6, r30, r30
	dcsel	16, r2, r29
	dcp	r30, pls.addr, north
	dcp	r29, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	cr11, north
	dstcr	0x200, pc.mode, north
.LBB34_72:                              //   Parent Loop BB34_1 Depth=1
                                        //     Parent Loop BB34_71 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB34_71 Depth=2
	djmpincsetup	0, 16, :.LBB34_74
	dstcr	0x300, pc.mode, north
.LBB34_74:                              //   Parent Loop BB34_1 Depth=1
                                        //     Parent Loop BB34_71 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.75:                              //   in Loop: Header=BB34_71 Depth=2
	daddi32	r28, 1, r28
	dstcr	1, r29
                                        // implicit-def: $cx11
	djmpeqoff	r28, r7, :.LBB34_77
// %bb.76:                              //   in Loop: Header=BB34_71 Depth=2
	cp	[crp4+=1], cr11
	dstcr	0, r29
.LBB34_77:                              // %Flow8
                                        //   in Loop: Header=BB34_71 Depth=2
	djmpeqoff	r29, 0, :.LBB34_71
.LBB34_78:                              // %.loopexit7
                                        //   in Loop: Header=BB34_1 Depth=1
	djmplt	r17, r15, :.LBB34_86
// %bb.79:                              //   in Loop: Header=BB34_1 Depth=1
	dshlb	r7, 8, r28
	dshlb	r7, 2, r7
	daddi32	r28, r17, r17
	addi32	crp1, 24, crp4          //      
	dshrab	r17, 31, r28
	dsubi32	r13, r17, r30
	dshrlb	r28, 28, r28
	daddi32	r17, r11, r29
	daddi32	r17, r28, r28
	dcmplti32	r17, r14, r17
	dshrab	r28, 4, r17
	dshrab	r30, 31, r28
	dshlb	r17, 6, r17
	dshrlb	r28, 28, r28
	daddi32	r6, r17, r2
	daddi32	r30, r28, r17
	dcpc	r7, crp5
	dshrab	r17, 4, r17
	dcp	r2, pls.addr, north
	dcsel	1, r17, r7
	dcmplt32	r29, r16, r28
	dcp	r7, pls.count1, north
	addi32	crp4, crp5, crp4
	dstcr	1, r6
	dcsel	1, r17, r17
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r17, :.LBB34_80
// %bb.93:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r5, cr11
	cmplti32	cr10, cr11, cr11
	predpush	cr11, :.LBB34_95
// %bb.94:                              //   in Loop: Header=BB34_1 Depth=1
	nrb	[crp4], north
.LBB34_95:                              //   in Loop: Header=BB34_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB34_96
	dstcr	0x200, pc.mode, north
.LBB34_96:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.97:                              //   in Loop: Header=BB34_1 Depth=1
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB34_98
.LBB34_98:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.99:                              // %Flow
                                        //   in Loop: Header=BB34_1 Depth=1
	dstcr	0, r6
.LBB34_80:                              // %Flow6
                                        //   in Loop: Header=BB34_1 Depth=1
	djmpeqoff	0, r6, :.LBB34_86
// %bb.81:                              //   in Loop: Header=BB34_1 Depth=1
	dcpc	r5, cr11
	cmplti32	cr10, cr11, cr10
	predpush	cr10, :.LBB34_83
// %bb.82:                              //   in Loop: Header=BB34_1 Depth=1
	nrb	[crp4], north
.LBB34_83:                              //   in Loop: Header=BB34_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB34_84
	dstcr	0x200, pc.mode, north
.LBB34_84:                              //   Parent Loop BB34_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.85:                              //   in Loop: Header=BB34_1 Depth=1
	dstcr	0x300, pc.mode, north
.LBB34_86:                              // %.loopexit
                                        //   in Loop: Header=BB34_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-371, r31
	djmpincne	r10, 10, r31
.LBB34_87:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r22
	dcp	[rp1 + 2], r21
	dcp	[rp1 + 4], r20
	dcp	[rp1 + 6], r19
	dcp	[rp2], r18
	daddi32	rp1, 40, rp2
	dcp	[rp2], r9
	daddi32	rp1, 48, rp2
	dcp	[rp2], r8
	daddi32	rp1, 56, rp1
	addi32	crp1, 48, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z12fused_add_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
_Z12fused_add_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_: // @_Z12fused_add_14I10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj1ELj1ELj2535EEES6_S6_EvRT0_RT1_RT2_
// %bb.0:
	daddi32	rp1, -56, rp1
	daddi32	rp1, 48, rp2
	dstcr	0x2, mode
	addi32	crp1, -48, crp1         //     
	dcp	r11, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 40, rp2
	dcp	r10, rp4
	dcp	r9, [rp2]
	daddi32	rp1, 32, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	dcp	r12, rp2
	dstcr	256, r11
	dstcr	2535, r12
	dstcr	2550, r13
	addi32	crp1, 24, crp2          //      
	dstcr	2280, r14
	cp	crp1, crp3
	dstcr	2279, r15
	dstcr	2536, r16
	dcp	r19, [rp1 + 6]
	dcp	r20, [rp1 + 4]
	dcp	r21, [rp1 + 2]
	dcp	r22, [rp1]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x9f0, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x9f0, pls.stride2, north
.LBB35_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB35_7 Depth 2
                                        //       Child Loop BB35_8 Depth 3
                                        //       Child Loop BB35_5 Depth 3
                                        //     Child Loop BB35_12 Depth 2
                                        //     Child Loop BB35_14 Depth 2
                                        //     Child Loop BB35_16 Depth 2
                                        //     Child Loop BB35_22 Depth 2
                                        //     Child Loop BB35_24 Depth 2
                                        //     Child Loop BB35_31 Depth 2
                                        //     Child Loop BB35_33 Depth 2
                                        //     Child Loop BB35_91 Depth 2
                                        //       Child Loop BB35_92 Depth 3
                                        //       Child Loop BB35_89 Depth 3
                                        //     Child Loop BB35_43 Depth 2
                                        //     Child Loop BB35_45 Depth 2
                                        //     Child Loop BB35_47 Depth 2
                                        //     Child Loop BB35_53 Depth 2
                                        //     Child Loop BB35_55 Depth 2
                                        //     Child Loop BB35_62 Depth 2
                                        //     Child Loop BB35_64 Depth 2
                                        //     Child Loop BB35_71 Depth 2
                                        //       Child Loop BB35_72 Depth 3
                                        //       Child Loop BB35_74 Depth 3
                                        //     Child Loop BB35_96 Depth 2
                                        //     Child Loop BB35_98 Depth 2
                                        //     Child Loop BB35_84 Depth 2
	dcmplt32	8, r10, r17
	dshlb	r10, 8, r17
	dcsel	r12, r11, r6
	dcmpneq32	r10, 9, r5
	dsubi32	r12, r17, r5
	dcp	[rp4], r29
	dshrlb	r5, 8, r5
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	dandb	r5, 255, r7
	shlb	row, 4, cr10
	dcsel	0, 231, r28
	cp	crp2, crp4
	dstcr	0, r30
	dcsel	1, r7, r7
	addi32	cr10, col, cr10
	dstcr	0x0, pc.constant, south
	djmpeqoff	0, r7, :.LBB35_2
.LBB35_7:                               // %.preheader14
                                        //   Parent Loop BB35_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB35_8 Depth 3
                                        //       Child Loop BB35_5 Depth 3
	dshlb	r30, 8, r2
	djmpincsetup	0, 16, :.LBB35_8
	daddi32	r2, r17, r2
	dsubi32	r13, r2, r9
	dshlb	r2, 2, r8
	dcmplt32	r2, r14, r2
	dshrab	r9, 31, r2
	daddi32	r29, r8, r8
	dshrlb	r2, 28, r2
	dcp	r8, pls.addr, south
	daddi32	r9, r2, r2
	dshrab	r2, 4, r2
	dcsel	16, r2, r2
	dcp	r2, pls.count1, south
	dcp	[rp4 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp4 + 1]
	dstcr	0x300, pc.mode, south
.LBB35_8:                               //   Parent Loop BB35_1 Depth=1
                                        //     Parent Loop BB35_7 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.4:                               //   in Loop: Header=BB35_7 Depth=2
	djmpincsetup	0, 4, :.LBB35_5
	dstcr	0x200, pc.mode, south
.LBB35_5:                               //   Parent Loop BB35_1 Depth=1
                                        //     Parent Loop BB35_7 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB35_7 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r30, r7, :.LBB35_7
.LBB35_2:                               // %.loopexit15
                                        //   in Loop: Header=BB35_1 Depth=1
	djmpneqoff	9, r10, :.LBB35_3
// %bb.9:                               //   in Loop: Header=BB35_1 Depth=1
	dshlb	r7, 8, r30
	dstcr	1, r2
	daddi32	r30, r17, r30
                                        // implicit-def: $cx11
	dsubi32	r6, r30, r8
	dshrlb	r30, 4, r22
	daddi32	r8, 15, r18
	dsubi32	r13, r30, r21
	dshrab	r18, 31, r20
	daddi32	r30, r11, r9
	dshrlb	r20, 28, r20
	dcmplt32	r30, r14, r30
	daddi32	r18, r20, r18
	dshlb	r22, 6, r20
	dshrab	r21, 31, r30
	daddi32	r29, r20, r29
	dandb	r8, 7, r19
	dcp	r29, pls.addr, south
	dshrlb	r30, 28, r29
	dandb	r18, -16, r18
	daddi32	r21, r29, r29
	dshrab	r29, 4, r29
	dcsel	1, r29, r29
	dcmpeq32	r19, 0, r30
	dcp	r29, pls.count1, south
	dcsel	r8, r18, r29
	dcmplt32	r6, r9, r30
	dshrab	r29, 31, r30
	dcp	[rp4 + 1], dependencyid
	dshrlb	r30, 28, r30
	dstcr	0x100, plsstatus, south
	daddi32	r29, r30, r29
	dcp	flowid, [rp4 + 1]
	dshrab	r29, 4, r29
	dcsel	r29, 16, r30
	dsubi32	16, r30, r29
	djmpeqoff	r30, 0, :.LBB35_29
// %bb.10:                              //   in Loop: Header=BB35_1 Depth=1
	dstcr	1, r2
                                        // implicit-def: $cx11
	dstcr	0x300, pc.mode, south
	djmpeqoff	0, r29, :.LBB35_20
// %bb.11:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r30, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB35_12
.LBB35_12:                              // %.preheader13
                                        //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.13:                              //   in Loop: Header=BB35_1 Depth=1
	djmpincsetup	0, 4, :.LBB35_14
	dstcr	0x200, pc.mode, south
.LBB35_14:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.15:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r29, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB35_16
.LBB35_16:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.17:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	predpush	cr12, :.LBB35_19
// %bb.18:                              //   in Loop: Header=BB35_1 Depth=1
	cp	south, cr11
.LBB35_19:                              // %Flow19
                                        //   in Loop: Header=BB35_1 Depth=1
	predpop	
	dstcr	0, r2
.LBB35_20:                              // %Flow21
                                        //   in Loop: Header=BB35_1 Depth=1
	djmpeqoff	0, r2, :.LBB35_28
// %bb.21:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r30, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB35_22
.LBB35_22:                              // %.preheader12
                                        //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.23:                              //   in Loop: Header=BB35_1 Depth=1
	djmpincsetup	0, 4, :.LBB35_24
	dstcr	0x200, pc.mode, south
.LBB35_24:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.25:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	dstcr	0x200, pc.mode, south
	predpush	cr12, :.LBB35_27
// %bb.26:                              //   in Loop: Header=BB35_1 Depth=1
	cp	south, cr11
.LBB35_27:                              // %Flow20
                                        //   in Loop: Header=BB35_1 Depth=1
	predpop	
.LBB35_28:                              // %Flow22
                                        //   in Loop: Header=BB35_1 Depth=1
	dstcr	0, r2
.LBB35_29:                              // %Flow24
                                        //   in Loop: Header=BB35_1 Depth=1
	djmpeqoff	r2, 0, :.LBB35_37
// %bb.30:                              //   in Loop: Header=BB35_1 Depth=1
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, 4, :.LBB35_31
	dstcr	0x200, pc.mode, south
.LBB35_31:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.32:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r29, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB35_33
.LBB35_33:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.34:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr10
	stcr	0, cr11
	predpush	cr10, :.LBB35_36
// %bb.35:                              //   in Loop: Header=BB35_1 Depth=1
	cp	south, cr11
.LBB35_36:                              // %Flow23
                                        //   in Loop: Header=BB35_1 Depth=1
	predpop	
.LBB35_37:                              //   in Loop: Header=BB35_1 Depth=1
	dshlb	r7, 2, r29
	dstcr	231, r28
	dcp	r5, r7
	dcpc	r29, crp4
	addi32	crp2, crp4, crp4
	cp	cr11, [crp4]
	djmp	:.LBB35_38
.LBB35_3:                               //   in Loop: Header=BB35_1 Depth=1
	dstcr	0, r28
	dstcr	1, r7
.LBB35_38:                              //   in Loop: Header=BB35_1 Depth=1
	dstcr	0x200, pc.mode, south
	dcp	[rp3], r29
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr10
	dstcr	0, r30
	cp	crp3, crp4
	addi32	cr10, col, cr10
	dstcr	0x0, pc.constant, south
	djmpeqoff	0, r7, :.LBB35_39
.LBB35_91:                              // %.preheader10
                                        //   Parent Loop BB35_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB35_92 Depth 3
                                        //       Child Loop BB35_89 Depth 3
	dshlb	r30, 8, r2
	djmpincsetup	0, 16, :.LBB35_92
	daddi32	r2, r17, r2
	dsubi32	r13, r2, r9
	dshlb	r2, 2, r8
	dcmplt32	r2, r14, r2
	dshrab	r9, 31, r2
	daddi32	r29, r8, r8
	dshrlb	r2, 28, r2
	dcp	r8, pls.addr, south
	daddi32	r9, r2, r2
	dshrab	r2, 4, r2
	dcsel	16, r2, r2
	dcp	r2, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB35_92:                              //   Parent Loop BB35_1 Depth=1
                                        //     Parent Loop BB35_91 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.88:                              //   in Loop: Header=BB35_91 Depth=2
	djmpincsetup	0, 4, :.LBB35_89
	dstcr	0x200, pc.mode, south
.LBB35_89:                              //   Parent Loop BB35_1 Depth=1
                                        //     Parent Loop BB35_91 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.90:                              //   in Loop: Header=BB35_91 Depth=2
	cp	south, [crp4+=1]
	djmpincne	r30, r7, :.LBB35_91
.LBB35_39:                              // %.loopexit11
                                        //   in Loop: Header=BB35_1 Depth=1
	djmpeqoff	0, r28, :.LBB35_69
// %bb.40:                              //   in Loop: Header=BB35_1 Depth=1
	dshlb	r7, 8, r2
	dstcr	1, r30
	daddi32	r2, r17, r2
                                        // implicit-def: $cx11
	dshrlb	r2, 4, r21
	dsubi32	r6, r2, r8
	dsubi32	r13, r2, r19
	dshlb	r21, 6, r21
	daddi32	r2, r11, r9
	dcmplti32	r2, r14, r2
	daddi32	r8, 15, r18
	daddi32	r29, r21, r29
	dshrab	r19, 31, r2
	dshrab	r18, 31, r20
	dcp	r29, pls.addr, south
	dshrlb	r2, 28, r29
	dshrlb	r20, 28, r20
	daddi32	r19, r29, r29
	daddi32	r18, r20, r18
	dshrab	r29, 4, r29
	dandb	r8, 7, r20
	dandb	r18, -16, r18
	dcsel	1, r29, r29
	dcmpeq32	r20, 0, r2
	dcp	r29, pls.count1, south
	dcsel	r8, r18, r29
	dcmplt32	r6, r9, r6
	dshrab	r29, 31, r6
	dcp	[rp3 + 1], dependencyid
	dshrlb	r6, 28, r6
	dstcr	0x100, plsstatus, south
	daddi32	r29, r6, r6
	dcp	flowid, [rp3 + 1]
	dshrab	r6, 4, r6
	dcsel	r6, 16, r29
	dsubi32	16, r29, r6
	djmpeqoff	r29, 0, :.LBB35_60
// %bb.41:                              //   in Loop: Header=BB35_1 Depth=1
	dstcr	1, r30
                                        // implicit-def: $cx11
	dstcr	0x300, pc.mode, south
	djmpeqoff	0, r6, :.LBB35_51
// %bb.42:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB35_43
.LBB35_43:                              // %.preheader9
                                        //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.44:                              //   in Loop: Header=BB35_1 Depth=1
	djmpincsetup	0, 4, :.LBB35_45
	dstcr	0x200, pc.mode, south
.LBB35_45:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.46:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB35_47
.LBB35_47:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.48:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	predpush	cr12, :.LBB35_50
// %bb.49:                              //   in Loop: Header=BB35_1 Depth=1
	cp	south, cr11
.LBB35_50:                              // %Flow10
                                        //   in Loop: Header=BB35_1 Depth=1
	predpop	
	dstcr	0, r30
.LBB35_51:                              // %Flow12
                                        //   in Loop: Header=BB35_1 Depth=1
	djmpeqoff	0, r30, :.LBB35_59
// %bb.52:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r29, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB35_53
.LBB35_53:                              // %.preheader8
                                        //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.54:                              //   in Loop: Header=BB35_1 Depth=1
	djmpincsetup	0, 4, :.LBB35_55
	dstcr	0x200, pc.mode, south
.LBB35_55:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.56:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	dstcr	0x200, pc.mode, south
	predpush	cr12, :.LBB35_58
// %bb.57:                              //   in Loop: Header=BB35_1 Depth=1
	cp	south, cr11
.LBB35_58:                              // %Flow11
                                        //   in Loop: Header=BB35_1 Depth=1
	predpop	
.LBB35_59:                              // %Flow13
                                        //   in Loop: Header=BB35_1 Depth=1
	dstcr	0, r30
.LBB35_60:                              // %Flow15
                                        //   in Loop: Header=BB35_1 Depth=1
	djmpeqoff	r30, 0, :.LBB35_68
// %bb.61:                              //   in Loop: Header=BB35_1 Depth=1
	dstcr	0x300, pc.mode, south
	djmpincsetup	0, 4, :.LBB35_62
	dstcr	0x200, pc.mode, south
.LBB35_62:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.63:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r6, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB35_64
.LBB35_64:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.65:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r28, cr11
	cmplti32	cr10, cr11, cr10
	stcr	0, cr11
	predpush	cr10, :.LBB35_67
// %bb.66:                              //   in Loop: Header=BB35_1 Depth=1
	cp	south, cr11
.LBB35_67:                              // %Flow14
                                        //   in Loop: Header=BB35_1 Depth=1
	predpop	
.LBB35_68:                              //   in Loop: Header=BB35_1 Depth=1
	dshlb	r7, 2, r6
	dcpc	r6, crp4
	addi32	crp3, crp4, crp4
	cp	cr11, [crp4]
.LBB35_69:                              //   in Loop: Header=BB35_1 Depth=1
	dstcr	0x200, pc.mode, south
	cp	[crp1 + 6], cr11
	cp	[crp1], cr12
	dcmplt32	r17, r15, r6
	addi32	cr12, cr11, [crp1 + 6]
	dcsel	1, r5, r7
	dcp	[rp2], r6
	shlb	row, 4, cr10
	dcsel	0, 231, r5
	addi32	cr10, col, cr10
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	djmpeqoff	r7, 0, :.LBB35_78
// %bb.70:                              //   in Loop: Header=BB35_1 Depth=1
	addi32	cr12, cr11, cr11
	orb	crp2, 4, crp4
	dstcr	0, r28
.LBB35_71:                              // %.preheader
                                        //   Parent Loop BB35_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB35_72 Depth 3
                                        //       Child Loop BB35_74 Depth 3
	dshlb	r28, 8, r29
	djmpincsetup	0, 4, :.LBB35_72
	daddi32	r29, r17, r29
	dshrab	r29, 31, r30
	dsubi32	r13, r29, r2
	dshrlb	r30, 28, r30
	dshrab	r2, 31, r8
	daddi32	r29, r30, r30
	dshrlb	r8, 28, r8
	dshlb	r30, 2, r30
	daddi32	r2, r8, r2
	dandb	r30, -64, r30
	dshrab	r2, 4, r2
	dcmplti32	r29, r14, r29
	daddi32	r6, r30, r30
	dcsel	16, r2, r29
	dcp	r30, pls.addr, north
	dcp	r29, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	cr11, north
	dstcr	0x200, pc.mode, north
.LBB35_72:                              //   Parent Loop BB35_1 Depth=1
                                        //     Parent Loop BB35_71 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.73:                              //   in Loop: Header=BB35_71 Depth=2
	djmpincsetup	0, 16, :.LBB35_74
	dstcr	0x300, pc.mode, north
.LBB35_74:                              //   Parent Loop BB35_1 Depth=1
                                        //     Parent Loop BB35_71 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.75:                              //   in Loop: Header=BB35_71 Depth=2
	daddi32	r28, 1, r28
	dstcr	1, r29
                                        // implicit-def: $cx11
	djmpeqoff	r28, r7, :.LBB35_77
// %bb.76:                              //   in Loop: Header=BB35_71 Depth=2
	cp	[crp4+=1], cr11
	dstcr	0, r29
.LBB35_77:                              // %Flow8
                                        //   in Loop: Header=BB35_71 Depth=2
	djmpeqoff	r29, 0, :.LBB35_71
.LBB35_78:                              // %.loopexit7
                                        //   in Loop: Header=BB35_1 Depth=1
	djmplt	r17, r15, :.LBB35_86
// %bb.79:                              //   in Loop: Header=BB35_1 Depth=1
	dshlb	r7, 8, r28
	dshlb	r7, 2, r7
	daddi32	r28, r17, r17
	addi32	crp1, 24, crp4          //      
	dshrab	r17, 31, r28
	dsubi32	r13, r17, r30
	dshrlb	r28, 28, r28
	daddi32	r17, r11, r29
	daddi32	r17, r28, r28
	dcmplti32	r17, r14, r17
	dshrab	r28, 4, r17
	dshrab	r30, 31, r28
	dshlb	r17, 6, r17
	dshrlb	r28, 28, r28
	daddi32	r6, r17, r2
	daddi32	r30, r28, r17
	dcpc	r7, crp5
	dshrab	r17, 4, r17
	dcp	r2, pls.addr, north
	dcsel	1, r17, r7
	dcmplt32	r29, r16, r28
	dcp	r7, pls.count1, north
	addi32	crp4, crp5, crp4
	dstcr	1, r6
	dcsel	1, r17, r17
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r17, :.LBB35_80
// %bb.93:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r5, cr11
	cmplti32	cr10, cr11, cr11
	predpush	cr11, :.LBB35_95
// %bb.94:                              //   in Loop: Header=BB35_1 Depth=1
	nrb	[crp4], north
.LBB35_95:                              //   in Loop: Header=BB35_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB35_96
	dstcr	0x200, pc.mode, north
.LBB35_96:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.97:                              //   in Loop: Header=BB35_1 Depth=1
	dcp	r17, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB35_98
.LBB35_98:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.99:                              // %Flow
                                        //   in Loop: Header=BB35_1 Depth=1
	dstcr	0, r6
.LBB35_80:                              // %Flow6
                                        //   in Loop: Header=BB35_1 Depth=1
	djmpeqoff	0, r6, :.LBB35_86
// %bb.81:                              //   in Loop: Header=BB35_1 Depth=1
	dcpc	r5, cr11
	cmplti32	cr10, cr11, cr10
	predpush	cr10, :.LBB35_83
// %bb.82:                              //   in Loop: Header=BB35_1 Depth=1
	nrb	[crp4], north
.LBB35_83:                              //   in Loop: Header=BB35_1 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB35_84
	dstcr	0x200, pc.mode, north
.LBB35_84:                              //   Parent Loop BB35_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.85:                              //   in Loop: Header=BB35_1 Depth=1
	dstcr	0x300, pc.mode, north
.LBB35_86:                              // %.loopexit
                                        //   in Loop: Header=BB35_1 Depth=1
	dstcr	0x200, pc.mode, north
	dstcr	-371, r31
	djmpincne	r10, 10, r31
.LBB35_87:
	daddi32	rp1, 32, rp2
	dcp	[rp1], r22
	dcp	[rp1 + 2], r21
	dcp	[rp1 + 4], r20
	dcp	[rp1 + 6], r19
	dcp	[rp2], r18
	daddi32	rp1, 40, rp2
	dcp	[rp2], r9
	daddi32	rp1, 48, rp2
	dcp	[rp2], r8
	daddi32	rp1, 56, rp1
	addi32	crp1, 48, crp1          //     
	dret	
                                        // -- End function
	.p2align	3               // -- Begin function _Z10fused_copyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj81ELj1ELj2535EEES6_EvRT0_RT1_
_Z10fused_copyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj81ELj1ELj2535EEES6_EvRT0_RT1_: // @_Z10fused_copyI10FixedPointIsLh6ELh2ELi0EE7_TensorIS0_IiLh16ELh4ELi0EEL10HasBorders0EjLj64EL14TensorLocation1EJLj1ELj81ELj1ELj2535EEES6_EvRT0_RT1_
// %bb.0:
	daddi32	rp1, -88, rp1
	daddi32	rp1, 80, rp2
	dstcr	0x2, mode
	stcr	-344, cr10
	dcp	r10, rp3
	dcp	r8, [rp2]
	daddi32	rp1, 72, rp2
	addi32	crp1, cr10, crp1        //     
	dcp	r9, [rp2]
	daddi32	rp1, 64, rp2
	dstcr	0, r10
	dcp	r18, [rp2]
	daddi32	rp1, 56, rp2
	dstcr	2535, r12
	dcp	r19, [rp2]
	daddi32	rp1, 48, rp2
	dstcr	2550, r13
	dcp	r20, [rp2]
	daddi32	rp1, 40, rp2
	dstcr	2280, r14
	dcp	r21, [rp2]
	daddi32	rp1, 32, rp2
	addi32	crp1, 16, crp2          //      
	dcp	r22, [rp2]
	dcp	r11, rp2
	dstcr	256, r11
	dstcr	10176, r15
	dstcr	2279, r16
	dstcr	2536, r17
	dcp	r23, [rp1 + 6]
	dcp	r24, [rp1 + 4]
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x9f0, pls.stride2, south
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	stcr	0x2, bitwidthmode
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x9f0, pls.stride2, north
.LBB36_1:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB36_2 Depth 2
                                        //       Child Loop BB36_4 Depth 3
                                        //         Child Loop BB36_5 Depth 4
                                        //         Child Loop BB36_7 Depth 4
                                        //       Child Loop BB36_13 Depth 3
                                        //       Child Loop BB36_15 Depth 3
                                        //       Child Loop BB36_17 Depth 3
                                        //       Child Loop BB36_23 Depth 3
                                        //       Child Loop BB36_25 Depth 3
                                        //       Child Loop BB36_32 Depth 3
                                        //       Child Loop BB36_34 Depth 3
                                        //     Child Loop BB36_41 Depth 2
                                        //       Child Loop BB36_43 Depth 3
                                        //         Child Loop BB36_44 Depth 4
                                        //         Child Loop BB36_46 Depth 4
                                        //       Child Loop BB36_62 Depth 3
                                        //       Child Loop BB36_64 Depth 3
                                        //       Child Loop BB36_54 Depth 3
	dcmplt32	8, r10, r5
	dshlb	r10, 8, r5
	dcsel	r12, r11, r2
	dsubi32	r12, r5, r6
	dcmpeq32	r10, 9, r19
	dshrlb	r6, 8, r6
	dcp	[rp3], r7
	dandb	r6, 255, r30
	dstcr	0x11, pls.mode, south
	dcsel	r30, 1, r30
	dstcr	0x300, pc.mode, south
	dshlb	r30, 8, r8
	dstcr	0x200, pc.mode, north
	daddi32	r8, r5, r9
	shlb	row, 4, cr10
	dsubi32	r2, r9, r8
	daddi32	r9, r11, r18
	daddi32	r8, 15, r20
	dandb	r8, 7, r22
	dshrab	r20, 31, r21
	dcmpeq32	r22, 0, r22
	dshrlb	r21, 28, r21
	dstcr	0, r28
	daddi32	r20, r21, r20
	dsubi32	r13, r9, r21
	dandb	r20, -16, r20
	dshrab	r21, 31, r22
	dcsel	r8, r20, r8
	dcmplt32	r2, r18, r2
	dshrab	r8, 31, r2
	dshrlb	r22, 28, r18
	dshrlb	r2, 28, r2
	daddi32	r21, r18, r18
	daddi32	r8, r2, r2
	dshrab	r18, 4, r18
	dshrab	r2, 4, r8
	dshrlb	r9, 4, r2
	dcsel	r8, 16, r8
	dcmplt32	r9, r14, r9
	dcsel	1, r18, r18
	dcmpneq32	r19, 0, r19
	dstcr	0, r29
	addi32	cr10, col, cr10
	dsubi32	16, r8, r9
	dcsel	231, 0, r19
	dstcr	0x0, pc.constant, south
.LBB36_2:                               //   Parent Loop BB36_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB36_4 Depth 3
                                        //         Child Loop BB36_5 Depth 4
                                        //         Child Loop BB36_7 Depth 4
                                        //       Child Loop BB36_13 Depth 3
                                        //       Child Loop BB36_15 Depth 3
                                        //       Child Loop BB36_17 Depth 3
                                        //       Child Loop BB36_23 Depth 3
                                        //       Child Loop BB36_25 Depth 3
                                        //       Child Loop BB36_32 Depth 3
                                        //       Child Loop BB36_34 Depth 3
	djmpeqoff	0, r30, :.LBB36_9
// %bb.3:                               //   in Loop: Header=BB36_2 Depth=2
	dshlb	r29, 2, r21
	dstcr	0, r20
	dcpc	r21, crp3
	addi32	crp2, crp3, crp3
.LBB36_4:                               // %.preheader10
                                        //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB36_5 Depth 4
                                        //         Child Loop BB36_7 Depth 4
	dshlb	r20, 8, r22
	dmuli32	r28, r15, r21
	daddi32	r22, r5, r22
	djmpincsetup	0, 16, :.LBB36_5
	daddi32	r7, r21, r21
	dshlb	r22, 2, r23
	dsubi32	r13, r22, r24
	daddi32	r21, r23, r21
	dshrab	r24, 31, r23
	dcmplt32	r22, r14, r22
	dshrlb	r23, 28, r22
	dcp	r21, pls.addr, south
	daddi32	r24, r22, r21
	dshrab	r21, 4, r21
	dcsel	16, r21, r21
	dcp	r21, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
.LBB36_5:                               //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        //       Parent Loop BB36_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.6:                               //   in Loop: Header=BB36_4 Depth=3
	djmpincsetup	0, 4, :.LBB36_7
	dstcr	0x200, pc.mode, south
.LBB36_7:                               //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        //       Parent Loop BB36_4 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.8:                               //   in Loop: Header=BB36_4 Depth=3
	cp	south, [crp3+=1]
	daddi32	r29, 1, r29
	djmpincne	r20, r30, :.LBB36_4
.LBB36_9:                               // %.loopexit11
                                        //   in Loop: Header=BB36_2 Depth=2
	djmpneqoff	9, r10, :.LBB36_39
// %bb.10:                              //   in Loop: Header=BB36_2 Depth=2
	dmuli32	r28, r15, r21
	dshlb	r2, 6, r22
	dstcr	1, r20
	daddi32	r7, r21, r21
                                        // implicit-def: $cx11
	daddi32	r21, r22, r21
	dcp	r21, pls.addr, south
	dcp	r18, pls.count1, south
	dcp	[rp3 + 1], dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, [rp3 + 1]
	dstcr	0x300, pc.mode, south
	djmpeqoff	r8, 0, :.LBB36_30
// %bb.11:                              //   in Loop: Header=BB36_2 Depth=2
	dstcr	1, r20
                                        // implicit-def: $cx11
	djmpeqoff	0, r9, :.LBB36_21
// %bb.12:                              //   in Loop: Header=BB36_2 Depth=2
	dcp	r8, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB36_13
.LBB36_13:                              // %.preheader9
                                        //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.14:                              //   in Loop: Header=BB36_2 Depth=2
	djmpincsetup	0, 4, :.LBB36_15
	dstcr	0x200, pc.mode, south
.LBB36_15:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.16:                              //   in Loop: Header=BB36_2 Depth=2
	dcp	r9, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB36_17
.LBB36_17:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.18:                              //   in Loop: Header=BB36_2 Depth=2
	dcpc	r19, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	predpush	cr12, :.LBB36_20
// %bb.19:                              //   in Loop: Header=BB36_2 Depth=2
	cp	south, cr11
.LBB36_20:                              // %Flow7
                                        //   in Loop: Header=BB36_2 Depth=2
	predpop	
	dstcr	0, r20
.LBB36_21:                              // %Flow9
                                        //   in Loop: Header=BB36_2 Depth=2
	djmpeqoff	0, r20, :.LBB36_29
// %bb.22:                              //   in Loop: Header=BB36_2 Depth=2
	dcp	r8, jumpendcount
	djmpincsetup	0, jumpendcount, :.LBB36_23
.LBB36_23:                              // %.preheader8
                                        //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.24:                              //   in Loop: Header=BB36_2 Depth=2
	djmpincsetup	0, 4, :.LBB36_25
	dstcr	0x200, pc.mode, south
.LBB36_25:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.26:                              //   in Loop: Header=BB36_2 Depth=2
	dcpc	r19, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	dstcr	0x200, pc.mode, south
	predpush	cr12, :.LBB36_28
// %bb.27:                              //   in Loop: Header=BB36_2 Depth=2
	cp	south, cr11
.LBB36_28:                              // %Flow8
                                        //   in Loop: Header=BB36_2 Depth=2
	predpop	
.LBB36_29:                              // %Flow10
                                        //   in Loop: Header=BB36_2 Depth=2
	dstcr	0, r20
.LBB36_30:                              // %Flow12
                                        //   in Loop: Header=BB36_2 Depth=2
	djmpeqoff	r20, 0, :.LBB36_38
// %bb.31:                              //   in Loop: Header=BB36_2 Depth=2
	djmpincsetup	0, 4, :.LBB36_32
	dstcr	0x200, pc.mode, south
.LBB36_32:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.33:                              //   in Loop: Header=BB36_2 Depth=2
	dcp	r9, jumpendcount
	dstcr	0x200, pc.mode, south
	djmpincsetup	0, jumpendcount, :.LBB36_34
.LBB36_34:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.35:                              //   in Loop: Header=BB36_2 Depth=2
	dcpc	r19, cr11
	cmplti32	cr10, cr11, cr12
	stcr	0, cr11
	predpush	cr12, :.LBB36_37
// %bb.36:                              //   in Loop: Header=BB36_2 Depth=2
	cp	south, cr11
.LBB36_37:                              // %Flow11
                                        //   in Loop: Header=BB36_2 Depth=2
	predpop	
.LBB36_38:                              //   in Loop: Header=BB36_2 Depth=2
	dshlb	r29, 2, r20
	daddi32	r29, 1, r29
	dcpc	r20, crp3
	addi32	crp2, crp3, crp3
	cp	cr11, [crp3]
.LBB36_39:                              //   in Loop: Header=BB36_2 Depth=2
	djmpincne	r28, 81, :.LBB36_2
// %bb.40:                              //   in Loop: Header=BB36_1 Depth=1
	dcmplt32	r5, r16, r8
	dcsel	1, r6, r6
	dstcr	0x200, pc.mode, south
	dshlb	r6, 8, r29
	dstcr	0, r7
	daddi32	r29, r5, r30
	dstcr	0, r28
	dsubi32	r13, r30, r29
	daddi32	r30, r11, r2
	dshrab	r29, 31, r9
	dcmplt32	r2, r17, r2
	dshrlb	r9, 28, r2
	dshrab	r30, 31, r9
	daddi32	r29, r2, r29
	dshrlb	r9, 28, r2
	dshrab	r29, 4, r9
	daddi32	r30, r2, r2
	dcsel	1, r9, r29
	dcmplti32	r30, r14, r30
	dcsel	1, r9, r30
	dcp	[rp2], r9
	shlb	row, 4, cr10
	dcmpneq32	r8, 0, r8
	dshrab	r2, 4, r2
	dcsel	0, 231, r8
	addi32	cr10, col, cr10
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
.LBB36_41:                              //   Parent Loop BB36_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB36_43 Depth 3
                                        //         Child Loop BB36_44 Depth 4
                                        //         Child Loop BB36_46 Depth 4
                                        //       Child Loop BB36_62 Depth 3
                                        //       Child Loop BB36_64 Depth 3
                                        //       Child Loop BB36_54 Depth 3
	djmpeqoff	r6, 0, :.LBB36_48
// %bb.42:                              //   in Loop: Header=BB36_41 Depth=2
	dshlb	r28, 2, r19
	dstcr	0, r18
	dcpc	r19, crp3
	addi32	crp2, crp3, crp3
.LBB36_43:                              // %.preheader
                                        //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_41 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB36_44 Depth 4
                                        //         Child Loop BB36_46 Depth 4
	dshlb	r18, 8, r19
	dmuli32	r7, r15, r20
	daddi32	r19, r5, r19
	djmpincsetup	0, 4, :.LBB36_44
	dshrab	r19, 31, r21
	dsubi32	r13, r19, r22
	dshrlb	r21, 28, r21
	daddi32	r9, r20, r20
	daddi32	r19, r21, r21
	dcmplti32	r19, r14, r19
	dshlb	r21, 2, r19
	dshrab	r22, 31, r21
	dandb	r19, -64, r19
	dshrlb	r21, 28, r21
	daddi32	r20, r19, r19
	daddi32	r22, r21, r20
	dcp	r19, pls.addr, north
	dshrab	r20, 4, r20
	dcsel	16, r20, r19
	dcp	r19, pls.count1, north
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	nrb	[crp3], north
	dstcr	0x200, pc.mode, north
.LBB36_44:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_41 Depth=2
                                        //       Parent Loop BB36_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.45:                              //   in Loop: Header=BB36_43 Depth=3
	djmpincsetup	0, 16, :.LBB36_46
	dstcr	0x300, pc.mode, north
.LBB36_46:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_41 Depth=2
                                        //       Parent Loop BB36_43 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.47:                              //   in Loop: Header=BB36_43 Depth=3
	addi32	crp3, 4, crp3
	daddi32	r28, 1, r28
	djmpincne	r18, r6, :.LBB36_43
.LBB36_48:                              // %.loopexit7
                                        //   in Loop: Header=BB36_41 Depth=2
	djmplt	r5, r16, :.LBB36_56
// %bb.49:                              //   in Loop: Header=BB36_41 Depth=2
	dmuli32	r7, r15, r18
	dshlb	r2, 6, r19
	dshlb	r28, 2, r20
	daddi32	r9, r18, r18
	daddi32	r28, 1, r28
	daddi32	r18, r19, r19
	dcpc	r20, crp3
	dcp	r19, pls.addr, north
	dcp	r30, pls.count1, north
	addi32	crp2, crp3, crp3
	dstcr	1, r18
	dcp	[rp2 + 1], dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, [rp2 + 1]
	djmpeqoff	0, r29, :.LBB36_50
// %bb.59:                              //   in Loop: Header=BB36_41 Depth=2
	dcpc	r8, cr11
	cmplti32	cr10, cr11, cr11
	predpush	cr11, :.LBB36_61
// %bb.60:                              //   in Loop: Header=BB36_41 Depth=2
	nrb	[crp3], north
.LBB36_61:                              //   in Loop: Header=BB36_41 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB36_62
	dstcr	0x200, pc.mode, north
.LBB36_62:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.63:                              //   in Loop: Header=BB36_41 Depth=2
	dcp	r29, jumpendcount
	dstcr	0x300, pc.mode, north
	djmpincsetup	0, jumpendcount, :.LBB36_64
.LBB36_64:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.65:                              // %Flow
                                        //   in Loop: Header=BB36_41 Depth=2
	dstcr	0, r18
.LBB36_50:                              // %Flow4
                                        //   in Loop: Header=BB36_41 Depth=2
	djmpeqoff	0, r18, :.LBB36_56
// %bb.51:                              //   in Loop: Header=BB36_41 Depth=2
	dcpc	r8, cr11
	cmplti32	cr10, cr11, cr11
	predpush	cr11, :.LBB36_53
// %bb.52:                              //   in Loop: Header=BB36_41 Depth=2
	nrb	[crp3], north
.LBB36_53:                              //   in Loop: Header=BB36_41 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB36_54
	dstcr	0x200, pc.mode, north
.LBB36_54:                              //   Parent Loop BB36_1 Depth=1
                                        //     Parent Loop BB36_41 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.55:                              //   in Loop: Header=BB36_41 Depth=2
	dstcr	0x300, pc.mode, north
.LBB36_56:                              // %.loopexit
                                        //   in Loop: Header=BB36_41 Depth=2
	djmpincne	r7, 81, :.LBB36_41
// %bb.57:                              //   in Loop: Header=BB36_1 Depth=1
	dstcr	0x200, pc.mode, north
	djmpincne	r10, 10, :.LBB36_1
// %bb.58:
	daddi32	rp1, 32, rp2
	dcp	[rp1 + 4], r24
	dcp	[rp1 + 6], r23
	dcp	[rp2], r22
	daddi32	rp1, 40, rp2
	dcp	[rp2], r21
	daddi32	rp1, 48, rp2
	dcp	[rp2], r20
	daddi32	rp1, 56, rp2
	dcp	[rp2], r19
	daddi32	rp1, 64, rp2
	dcp	[rp2], r18
	daddi32	rp1, 72, rp2
	dcp	[rp2], r9
	daddi32	rp1, 80, rp2
	dcp	[rp2], r8
	daddi32	rp1, 88, rp1
	stcr	344, cr10
	addi32	crp1, cr10, crp1        //     
	dret	
                                        // -- End function
	.rodata.str1.4
	.p2align	2               // @.str
.L.str:
	.zero	1


	.note.GNU-stack
