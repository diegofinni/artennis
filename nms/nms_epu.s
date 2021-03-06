	.text
	.hidden	filter_boxes            // -- Begin function filter_boxes
	.globl	filter_boxes
	.p2align	3
filter_boxes:                           // @filter_boxes
// %bb.0:                               // %.preheader82.preheader
	dstcr	0, rp0
	dstcr	0, rp1
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
	daddi32	rp1, -32, rp1
	stcr	-3104, cr10
	stcr	2732, crp2
	addi32	crp1, cr10, crp1        //     
	dcp	p11, r16
	addi32	crp1, crp2, crp2
	dcp	p10, r17
	dcp	p9, r12
	dcp	p8, r13
	dcp	p7, r14
	dcp	p6, r15
	dcp	p5, r10
	dcp	p4, r11
	dcp	p3, r5
	dcp	p2, r7
	dcp	p1, r6
	dcp	p0, r28
	addi32	crp2, 8, crp3
	djmpincsetup	0, 4, :.LBB0_1
	stcr	0x2, bitwidthmode
	cp	crp3, [crp1 + 6]
	dstcr	0x2, mode
	dstcr	4, [rp1 + 7]
	dstcr	3, [rp1 + 6]
	dstcr	1, [rp1 + 5]
	dstcr	2, [rp1 + 4]
	dstcr	0, [rp1 + 3]
	//APP
	dstcr 0, cyclecount
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "objectness filtering start"
	//NO_APP
	stcr	0x2, bitwidthmode
.LBB0_1:                                // %loadstoreloop6
                                        // =>This Inner Loop Header: Depth=1
	stcr.lb	0, [crp2+=1]
// %bb.2:                               // %split5
	addi32	crp1, 40, crp2          //      
	nop <> __iss__ print	r17 -fx12 -msg "confidencethreshFP"
	djmpincsetup	0, 10, :.LBB0_3
.LBB0_3:                                // %loadstoreloop4
                                        // =>This Inner Loop Header: Depth=1
	stcr.lb	0, [crp2+=1]
// %bb.4:                               // %split3
	stcr	2732, crp2
	dstcr	0, r30
	addi32	crp1, crp2, crp2
	dstcr	640, r2
	addi32	crp2, 12, crp3
	addi32	crp2, 4, crp2
	cp	crp3, [crp1 + 5]
	dstcr	0, r29
	cp	crp2, [crp1 + 7]
	addi32	crp1, 40, crp2          //      
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xa, pls.count1, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xa0, pls.stride2, north
.LBB0_5:                                // %.preheader82
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_6 Depth 2
                                        //       Child Loop BB0_9 Depth 3
                                        //       Child Loop BB0_11 Depth 3
	dcmpeq32	r30, 4, r8
	dsubi32	4, r30, r8
	dmuli32	r30, r2, r18
	shlb	row, 4, cr10
	daddi32	r30, 1, r30
	dcsel	r8, 1, r19
	addi32	cr10, col, cr10
	dcmplt32	r30, 4, r9
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dcp	r18, pls.addr, north
	dcp	r19, pls.count2, north
	dcp	r29, dependencyid
	dstcr	0x1, plsstatus, north
	cp	crp2, crp3
	dcsel	1, r8, r8
	dstcr	0, r9
	dcp	flowid, r29
	djmpeqoff	r8, 0, :.LBB0_13
.LBB0_6:                                // %.preheader80
                                        //   Parent Loop BB0_5 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_9 Depth 3
                                        //       Child Loop BB0_11 Depth 3
	cmplti32	cr10, 160, cr11
	predpush	cr11, :.LBB0_8
// %bb.7:                               //   in Loop: Header=BB0_6 Depth=2
	nrb	[crp3], north
.LBB0_8:                                //   in Loop: Header=BB0_6 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB0_9
	dstcr	0x200, pc.mode, north
.LBB0_9:                                //   Parent Loop BB0_5 Depth=1
                                        //     Parent Loop BB0_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.10:                              //   in Loop: Header=BB0_6 Depth=2
	djmpincsetup	0, 10, :.LBB0_11
	dstcr	0x300, pc.mode, north
.LBB0_11:                               //   Parent Loop BB0_5 Depth=1
                                        //     Parent Loop BB0_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.12:                              //   in Loop: Header=BB0_6 Depth=2
	addi32	crp3, 4, crp3
	djmpincne	r9, r8, :.LBB0_6
.LBB0_13:                               // %.loopexit81
                                        //   in Loop: Header=BB0_5 Depth=1
	dstcr	0x200, pc.mode, north
	djmpneqoff	r30, 5, :.LBB0_5
// %bb.14:
	dstcr	0x0, dependencyid
	dstcr	0x2, mode
	dstcr	3, r30
	dcp	p30, r2
	dandb	r2, 255, r2
.LBB0_15:                               // =>This Inner Loop Header: Depth=1
	dstcr	1, r8
	dcp	pelsr, r9
	djmpeqoff	r9, 0, :.LBB0_17
// %bb.16:                              //   in Loop: Header=BB0_15 Depth=1
	daddi32	r9, -2, r8
	dcmpeq32	r9, 1, r9
	dshlb	r30, r8, r8
	dcsel	1, r8, r8
	dandb	r2, r8, r8
	dcmpeq32	r8, 0, r8
.LBB0_17:                               // %Flow66
                                        //   in Loop: Header=BB0_15 Depth=1
	djmpeqoff	r8, 0, :.LBB0_15
// %bb.18:
	dstcr	0x7480, els.intaddr
	dcp	r28, els.extaddrl
	dcp	r6, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0xd32c0, els.intstride2
	dstcr	0x34cb, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0xd32c0, els.extstride2
	dstcr	0x34cb, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x1, els.mode
	dstcr	0x1, elsstatus
	dcp	flowid, r6
	dcp	flowid, r28
	dcp	p30, r28
	addi32	crp1, 40, crp2          //      
	dshlb	r28, 1, r28
	dstcr	0, r30
	dandb	r28, 254, r28
	dcp	r28, p30
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	cp	row, cr11
	cp	col, cr10
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x11380, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x9f, pls.count1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x9f0, pls.stride2, south
	dcp	r6, dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, r28
	dstcr	0x300, pc.mode, south
.LBB0_19:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_20 Depth 2
                                        //     Child Loop BB0_22 Depth 2
	djmpincsetup	0, 16, :.LBB0_20
.LBB0_20:                               //   Parent Loop BB0_19 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.21:                              //   in Loop: Header=BB0_19 Depth=1
	djmpincsetup	0, 4, :.LBB0_22
	dstcr	0x200, pc.mode, south
.LBB0_22:                               //   Parent Loop BB0_19 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.23:                              //   in Loop: Header=BB0_19 Depth=1
	cp	south, [crp2+=1]
	dstcr	0x300, pc.mode, south
	djmpincne	r30, 9, :.LBB0_19
// %bb.24:
	shlb	cr11, 4, cr11
	djmpincsetup	0, 15, :.LBB0_25
.LBB0_25:                               // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.26:
	addi32	cr10, cr11, cr10
	djmpincsetup	0, 4, :.LBB0_27
	dstcr	0x200, pc.mode, south
.LBB0_27:                               // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.28:
	cmplti32	cr10, 231, cr11
	stcr	0, cr10
	dstcr	0x200, pc.mode, south
	nnb	south, north
	predpush	cr11, :.LBB0_30
// %bb.29:
	cp	south, cr10
.LBB0_30:
	predpop	
	addi32	crp1, 40, crp2          //      
	djmpincsetup	0, 10, :.LBB0_31
	addi32	crp2, 36, crp3
	cp	cr10, [crp3]
	dstcr	0x200, pc.mode, south
.LBB0_31:                               // =>This Inner Loop Header: Depth=1
	shrab.lb	[crp2], 4, [crp2+=1]
// %bb.32:
	stcr	0, cr10
	dstcr	0, r6
	addi32	crp1, 40, crp2          //      
	stcr	256, cr11
	dstcr	0, r30
	dstcr	0x7200, pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
.LBB0_33:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_134 Depth 2
                                        //       Child Loop BB0_171 Depth 3
                                        //     Child Loop BB0_143 Depth 2
                                        //       Child Loop BB0_351 Depth 3
                                        //     Child Loop BB0_153 Depth 2
                                        //     Child Loop BB0_155 Depth 2
                                        //     Child Loop BB0_157 Depth 2
                                        //     Child Loop BB0_159 Depth 2
                                        //     Child Loop BB0_162 Depth 2
                                        //     Child Loop BB0_166 Depth 2
	cp	[crp2], cr13
	dcpc	r17, cr12
	cmplti32	cr13, cr12, cr14
	nrb	cr14, east
	nrb	cr14, west
	cmplti32	0, col, cr15
	cp	cr14, cr12
	predpush	cr15, :.LBB0_132
// %bb.34:                              //   in Loop: Header=BB0_33 Depth=1
	addi32	west, cr14, cr12
.LBB0_132:                              //   in Loop: Header=BB0_33 Depth=1
	predpop	
	cmplti32	col, 15, cr15
	predpush	cr15, :.LBB0_133
// %bb.170:                             //   in Loop: Header=BB0_33 Depth=1
	addi32	east, cr14, cr14
.LBB0_133:                              // %.preheader155
                                        //   in Loop: Header=BB0_33 Depth=1
	predpop	
	dstcr	2, r8
.LBB0_134:                              //   Parent Loop BB0_33 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_171 Depth 3
	dcp	r8, r2
	dstcr	0, r9
	daddi32	r2, -1, r8
	nrb	cr12, east
	nrb	cr14, west
	djmpltei	r8, 0, :.LBB0_135
.LBB0_171:                              // %.preheader77
                                        //   Parent Loop BB0_33 Depth=1
                                        //     Parent Loop BB0_134 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	daddi32	r9, 1, r9
	nnbr	r180
	dshlb	r9, 24, r18
	dshrab	r18, 24, r18
	djmplti	r18, r8, :.LBB0_171
.LBB0_135:                              // %.loopexit78
                                        //   in Loop: Header=BB0_134 Depth=2
	dcpc	r2, cr15
	cmpltei32	cr15, col, cr15
	predpush	cr15, :.LBB0_137
// %bb.136:                             //   in Loop: Header=BB0_134 Depth=2
	addi32	west, cr12, cr12
.LBB0_137:                              //   in Loop: Header=BB0_134 Depth=2
	predpop	
	dsubi32	15, r2, r8
	dcpc	r8, cr15
	cmpltei32	col, cr15, cr15
	predpush	cr15, :.LBB0_139
// %bb.138:                             //   in Loop: Header=BB0_134 Depth=2
	addi32	east, cr14, cr14
.LBB0_139:                              //   in Loop: Header=BB0_134 Depth=2
	predpop	
	dshlb	r2, 1, r8
	djmpltei	r2, 7, :.LBB0_134
// %bb.140:                             //   in Loop: Header=BB0_33 Depth=1
	cmplti32	0, cr14, cr16
	cmplti32	0, cr12, cr17
	dcpc	r17, cr15
	cmplti32	cr13, cr15, cr15
	andb	cr17, cr16, cr16
	addi32	cr14, cr12, cr14
	andb	cr15, cr16, cr15
	subi32	cr14, cr15, cr14
	nrb	cr14, south
	cmplti32	0, row, cr15
	predpush	cr15, :.LBB0_142
// %bb.141:                             //   in Loop: Header=BB0_33 Depth=1
	addi32	north, cr14, cr14
.LBB0_142:                              //   in Loop: Header=BB0_33 Depth=1
	predpop	
	dstcr	2, r8
	nrb	cr14, south
.LBB0_143:                              //   Parent Loop BB0_33 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_351 Depth 3
	dcp	r8, r2
	dstcr	0, r9
	daddi32	r2, -1, r8
	djmpltei	r8, 0, :.LBB0_144
.LBB0_351:                              // %.preheader75
                                        //   Parent Loop BB0_33 Depth=1
                                        //     Parent Loop BB0_143 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	daddi32	r9, 1, r9
	nnbr	r180
	dshlb	r9, 24, r18
	dshrab	r18, 24, r18
	djmplti	r18, r8, :.LBB0_351
.LBB0_144:                              // %.loopexit76
                                        //   in Loop: Header=BB0_143 Depth=2
	dcpc	r2, cr15
	cmpltei32	cr15, row, cr15
	predpush	cr15, :.LBB0_146
// %bb.145:                             //   in Loop: Header=BB0_143 Depth=2
	addi32	north, cr14, cr14
.LBB0_146:                              //   in Loop: Header=BB0_143 Depth=2
	predpop	
	dshlb	r2, 1, r8
	nrb	cr14, south
	djmpltei	r2, 7, :.LBB0_143
// %bb.147:                             //   in Loop: Header=BB0_33 Depth=1
	cmplti32	0, row, cr14
	predpush	cr14, :.LBB0_151
// %bb.148:                             //   in Loop: Header=BB0_33 Depth=1
	cmplti32	col, 16, cr14
	predpush	cr14, :.LBB0_150
// %bb.149:                             //   in Loop: Header=BB0_33 Depth=1
	addi32	north, cr12, cr12
.LBB0_150:                              // %Flow63
                                        //   in Loop: Header=BB0_33 Depth=1
	predpop	
.LBB0_151:                              //   in Loop: Header=BB0_33 Depth=1
	predpop	
	dcpc	r17, cr14
	cmpltei32	cr14, cr13, cr15
	subi32	cr10, cr12, cr13
	dshlb	r6, 8, r2
	addi32	cr13, col, cr13
	shlb	row, 4, cr14
	shlb	row, 4, cr16
	addi32	cr13, cr14, cr14
	dcpc	r2, cr13
	addi32	cr16, cr13, cr13
	cmplt32	cr14, 160, cr16
	addi32	cr13, col, cr13
	andb	cr16, cr15, cr15
	predpush	cr15, :.LBB0_161
// %bb.152:                             //   in Loop: Header=BB0_33 Depth=1
	dstcr	0x14, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	r30, dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, r30
	shlb	cr14, 2, cr14
	djmpincsetup	0, 4, :.LBB0_153
	nrb	cr14, north
	dstcr	0x260, pc.mode, north
.LBB0_153:                              //   Parent Loop BB0_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.154:                             //   in Loop: Header=BB0_33 Depth=1
	djmpincsetup	0, 16, :.LBB0_155
	dstcr	0x360, pc.mode, north
.LBB0_155:                              //   Parent Loop BB0_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.156:                             //   in Loop: Header=BB0_33 Depth=1
	djmpincsetup	0, 4, :.LBB0_157
	dstcr	0x260, pc.mode, north
	nrb	cr13, north
.LBB0_157:                              //   Parent Loop BB0_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.158:                             //   in Loop: Header=BB0_33 Depth=1
	djmpincsetup	0, 16, :.LBB0_159
	dstcr	0x360, pc.mode, north
.LBB0_159:                              //   Parent Loop BB0_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.160:                             //   in Loop: Header=BB0_33 Depth=1
	dstcr	0x260, pc.mode, north
.LBB0_161:                              // %Flow62
                                        //   in Loop: Header=BB0_33 Depth=1
	predpop	
	addi32	cr10, cr11, cr10
	dstcr	0, r2
	subi32	cr10, cr12, cr10
.LBB0_162:                              //   Parent Loop BB0_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nrb	cr10, north
	cmplti32	row, 15, cr12
	predpush	cr12, :.LBB0_164
// %bb.163:                             //   in Loop: Header=BB0_162 Depth=2
	cp	south, cr10
.LBB0_164:                              //   in Loop: Header=BB0_162 Depth=2
	predpop	
	daddi32	r2, 1, r2
	dandb	r2, 255, r8
	djmplte	r8, 14, :.LBB0_162
// %bb.165:                             //   in Loop: Header=BB0_33 Depth=1
	dstcr	0, r2
.LBB0_166:                              // %.preheader79
                                        //   Parent Loop BB0_33 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nrb	cr10, west
	cmplti32	col, 15, cr12
	predpush	cr12, :.LBB0_168
// %bb.167:                             //   in Loop: Header=BB0_166 Depth=2
	cp	east, cr10
.LBB0_168:                              //   in Loop: Header=BB0_166 Depth=2
	predpop	
	daddi32	r2, 1, r2
	dandb	r2, 255, r8
	djmplte	r8, 14, :.LBB0_166
// %bb.169:                             //   in Loop: Header=BB0_33 Depth=1
	addi32	crp2, 4, crp2
	djmpincne	r6, 10, :.LBB0_33
// %bb.35:
	//APP
	nop <> __iss__ profile stop -msg "objectness filtering end"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "store valid objects start"
	//NO_APP
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x6e00, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x100, pls.stride2, north
	dstcr	0x0, dependencyid
	dstcr	0x1, plsstatus, north
	dcp	flowid, r6
	djmpincsetup	0, 4, :.LBB0_36
	nrb	cr10, north
	dstcr	0x200, pc.mode, north
.LBB0_36:                               // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.37:
	djmpincsetup	0, 16, :.LBB0_38
	dstcr	0x300, pc.mode, north
.LBB0_38:                               // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.39:
	dstcr	0, r6
	dstcr	0x200, pc.mode, north
.LBB0_40:                               // =>This Inner Loop Header: Depth=1
	dcpc	r6, cr11
	cmplt32	cr11, cr10, cr11
	dstcr	1, r9
	predpush	cr11, :.LBB0_42
// %bb.41:                              //   in Loop: Header=BB0_40 Depth=1
	dstcr	0, r9
.LBB0_42:                               //   in Loop: Header=BB0_40 Depth=1
	predpop	
	dstcr	1, r2
	dstcr	1, r8
	djmpneqoff	0, r9, :.LBB0_43
// %bb.44:                              // %Landing27
                                        //   in Loop: Header=BB0_40 Depth=1
	djmpneqoff	0, r8, :.LBB0_45
.LBB0_46:                               // %Flow61
                                        //   in Loop: Header=BB0_40 Depth=1
	djmpeqoff	r2, 0, :.LBB0_40
	djmp	:.LBB0_47
.LBB0_43:                               // %Break29
                                        //   in Loop: Header=BB0_40 Depth=1
	dstcr	0, r8
	djmpeqoff	r8, 0, :.LBB0_46
.LBB0_45:                               //   in Loop: Header=BB0_40 Depth=1
	daddi32	r6, 1, r6
	dcmplt32	159, r6, r2
	djmpeqoff	r2, 0, :.LBB0_40
.LBB0_47:
	//APP
	nop <> __iss__ profile stop -msg "store valid objects end"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "copy filtered boxes and scores start"
	//NO_APP
	dstcr	0x11, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	cp	row, cr12
	cp	col, cr11
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x7200, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0xa, pls.count1, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0xa0, pls.stride2, south
	dcp	r30, dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, r30
	djmpincsetup	0, 10, :.LBB0_48
	dstcr	0x300, pc.mode, south
.LBB0_48:                               // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.49:
	djmpincsetup	0, 4, :.LBB0_50
	dstcr	0x200, pc.mode, south
.LBB0_50:                               // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.51:
	shlb	cr12, 4, cr12
	djmpincsetup	0, 6, :.LBB0_52
	dstcr	0x200, pc.mode, south
.LBB0_52:                               // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.53:
	addi32	cr11, cr12, cr12
	stcr	0, cr11
	cmplti32	cr12, 160, cr12
	predpush	cr12, :.LBB0_55
// %bb.54:
	stcr	0x2, bitwidthmode
	cp	south, cr11
.LBB0_55:
	predpop	
	dstcr	0x200, pc.mode, south
	stcr	2732, crp2
	shlb	row, 4, cr13
	dstcr	0, r30
	daddi32	rp1, 12, rp2
	addi32	crp1, crp2, crp2
	dstcr	2544, r2
	stcr	2535, cr12
	dstcr	33686018, r8
	addi32	cr13, col, cr13
	dstcr	0x7480, pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x2, mode
	stcr	0x2, bitwidthmode
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
.LBB0_56:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_59 Depth 2
                                        //     Child Loop BB0_61 Depth 2
                                        //     Child Loop BB0_63 Depth 2
                                        //     Child Loop BB0_65 Depth 2
                                        //     Child Loop BB0_67 Depth 2
	cmplt32	cr13, cr10, cr15
	stcr	0, cr14
	predpush	cr15, :.LBB0_70
// %bb.57:                              //   in Loop: Header=BB0_56 Depth=1
	cmplt32	cr11, cr12, cr16
	dmuli32	[rp2], r2, r9
	stcr	0, cr15
	dcpc	r9, cr14
	addi32	cr14, cr11, cr14
	predpush	cr16, :.LBB0_69
// %bb.58:                              //   in Loop: Header=BB0_56 Depth=1
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	r28, dependencyid
.LBB0_59:                               //   Parent Loop BB0_56 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dandb	r8, plsstatus, r9
	dorb	r9, pelsr, r9
	djmpneqoff	r9, 0, :.LBB0_59
// %bb.60:                              //   in Loop: Header=BB0_56 Depth=1
	shlb	cr14, 2, cr14
	djmpincsetup	0, 4, :.LBB0_61
	dstcr	0x1, plsstatus, north
	nrb	cr14, north
	dstcr	0x260, pc.mode, north
.LBB0_61:                               //   Parent Loop BB0_56 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.62:                              //   in Loop: Header=BB0_56 Depth=1
	djmpincsetup	0, 16, :.LBB0_63
	dstcr	0x360, pc.mode, north
.LBB0_63:                               //   Parent Loop BB0_56 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.64:                              //   in Loop: Header=BB0_56 Depth=1
	djmpincsetup	0, 16, :.LBB0_65
.LBB0_65:                               // %.preheader74
                                        //   Parent Loop BB0_56 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.66:                              //   in Loop: Header=BB0_56 Depth=1
	djmpincsetup	0, 4, :.LBB0_67
	dstcr	0x260, pc.mode, north
.LBB0_67:                               //   Parent Loop BB0_56 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.68:                              //   in Loop: Header=BB0_56 Depth=1
	cp	north, cr15
.LBB0_69:                               // %Flow59
                                        //   in Loop: Header=BB0_56 Depth=1
	predpop	
	shrab	cr15, 4, cr14
.LBB0_70:                               // %Flow60
                                        //   in Loop: Header=BB0_56 Depth=1
	predpop	
	daddi32	rp2, 4, rp2
	cp	cr14, [crp2+=1]
	djmpincne	r30, 4, :.LBB0_56
// %bb.71:
	shlb	row, 4, cr12
	stcr	2732, crp2
	addi32	cr12, col, cr12
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x0, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xa, pls.count1, north
	dstcr	0x4, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xa0, pls.stride2, north
	dcp	r29, dependencyid
	dstcr	0x1, plsstatus, north
	dstcr	0, r30
	addi32	crp1, crp2, crp2
	dcp	flowid, r29
.LBB0_72:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_75 Depth 2
                                        //     Child Loop BB0_77 Depth 2
	cmplti32	cr12, 160, cr13
	predpush	cr13, :.LBB0_74
// %bb.73:                              //   in Loop: Header=BB0_72 Depth=1
	nrb	[crp2], north
.LBB0_74:                               //   in Loop: Header=BB0_72 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB0_75
	dstcr	0x200, pc.mode, north
.LBB0_75:                               //   Parent Loop BB0_72 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.76:                              //   in Loop: Header=BB0_72 Depth=1
	djmpincsetup	0, 10, :.LBB0_77
	dstcr	0x300, pc.mode, north
.LBB0_77:                               //   Parent Loop BB0_72 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.78:                              //   in Loop: Header=BB0_72 Depth=1
	addi32	crp2, 4, crp2
	djmpincne	r30, 4, :.LBB0_72
// %bb.79:
	dstcr	0x200, pc.mode, north
	dstcr	0x7480, pls.addr, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	stcr	12720, cr12
	shlb	row, 4, cr14
	dstcr	0, r30
	addi32	cr11, cr12, cr12
	addi32	crp1, 40, crp2          //      
	dstcr	2544, r2
	stcr	2535, cr13
	dstcr	33686018, r8
	addi32	cr14, col, cr14
	dstcr	0x0, plsthresholdnorth
.LBB0_80:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_83 Depth 2
                                        //     Child Loop BB0_85 Depth 2
                                        //     Child Loop BB0_87 Depth 2
                                        //     Child Loop BB0_89 Depth 2
                                        //     Child Loop BB0_91 Depth 2
	cmplt32	cr14, cr10, cr16
	stcr	0, cr15
	predpush	cr16, :.LBB0_94
// %bb.81:                              //   in Loop: Header=BB0_80 Depth=1
	dmuli32	r30, r2, r9
	cmplt32	cr11, cr13, cr17
	stcr	0, cr16
	dcpc	r9, cr15
	addi32	cr12, cr15, cr15
	predpush	cr17, :.LBB0_93
// %bb.82:                              //   in Loop: Header=BB0_80 Depth=1
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	r28, dependencyid
.LBB0_83:                               //   Parent Loop BB0_80 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dandb	r8, plsstatus, r9
	dorb	r9, pelsr, r9
	djmpneqoff	r9, 0, :.LBB0_83
// %bb.84:                              //   in Loop: Header=BB0_80 Depth=1
	shlb	cr15, 2, cr15
	djmpincsetup	0, 4, :.LBB0_85
	dstcr	0x1, plsstatus, north
	nrb	cr15, north
	dstcr	0x260, pc.mode, north
.LBB0_85:                               //   Parent Loop BB0_80 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.86:                              //   in Loop: Header=BB0_80 Depth=1
	djmpincsetup	0, 16, :.LBB0_87
	dstcr	0x360, pc.mode, north
.LBB0_87:                               //   Parent Loop BB0_80 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.88:                              //   in Loop: Header=BB0_80 Depth=1
	djmpincsetup	0, 16, :.LBB0_89
.LBB0_89:                               // %.preheader73
                                        //   Parent Loop BB0_80 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.90:                              //   in Loop: Header=BB0_80 Depth=1
	djmpincsetup	0, 4, :.LBB0_91
	dstcr	0x260, pc.mode, north
.LBB0_91:                               //   Parent Loop BB0_80 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.92:                              //   in Loop: Header=BB0_80 Depth=1
	stcr	0x2, bitwidthmode
	cp	north, cr16
.LBB0_93:                               // %Flow57
                                        //   in Loop: Header=BB0_80 Depth=1
	predpop	
	shrlb	cr16, 4, cr15
.LBB0_94:                               // %Flow58
                                        //   in Loop: Header=BB0_80 Depth=1
	predpop	
	stcr	0x1, bitwidthmode
	cp	cr15, [crp2.z+=1]
	djmpincne	r30, 80, :.LBB0_80
// %bb.95:
	shlb	row, 4, cr11
	dstcr	0, r30
	addi32	cr11, col, cr11
	dstcr	0x8, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0xa00, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0xa, pls.count1, north
	dstcr	0x50, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0xa0, pls.stride2, north
	dstcr	0x0, dependencyid
	dstcr	0x1, plsstatus, north
	addi32	crp1, 40, crp2          //      
	dcp	flowid, r28
.LBB0_96:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_99 Depth 2
                                        //     Child Loop BB0_101 Depth 2
	cmplti32	cr11, 160, cr12
	predpush	cr12, :.LBB0_98
// %bb.97:                              //   in Loop: Header=BB0_96 Depth=1
	nrb	[crp2.z], north
.LBB0_98:                               //   in Loop: Header=BB0_96 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB0_99
	dstcr	0x200, pc.mode, north
.LBB0_99:                               //   Parent Loop BB0_96 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.100:                             //   in Loop: Header=BB0_96 Depth=1
	djmpincsetup	0, 10, :.LBB0_101
	dstcr	0x300, pc.mode, north
.LBB0_101:                              //   Parent Loop BB0_96 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.102:                             //   in Loop: Header=BB0_96 Depth=1
	addi32	crp2, 2, crp2
	djmpincne	r30, 80, :.LBB0_96
// %bb.103:
	dshlb	r29, 16, r30
	dstcr	32768, r2
	dshrab	r30, 31, r30
	dstcr	0x200, pc.mode, north
	dandb	r30, r29, r30
	dandb	r29, r2, r2
	dcp	r30, dependencyid
	dcp	p30, r30
	dshrlb	r2, 15, r9
	dstcr	3, r29
	dandb	r30, 255, r30
.LBB0_104:                              // =>This Inner Loop Header: Depth=1
	dstcr	1, r2
	dcp	pelsr, r8
	djmpeqoff	r8, 0, :.LBB0_106
// %bb.105:                             //   in Loop: Header=BB0_104 Depth=1
	daddi32	r8, -2, r2
	dcmpeq32	r8, 1, r8
	dshlb	r29, r2, r2
	dcsel	1, r2, r2
	dandb	r30, r2, r2
	dcmpeq32	r2, 0, r2
.LBB0_106:                              // %Flow56
                                        //   in Loop: Header=BB0_104 Depth=1
	djmpeqoff	r2, 0, :.LBB0_104
// %bb.107:
	dstcr	0x0, els.intaddr
	dcp	r7, els.extaddrl
	dcp	r5, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0xa00, els.intstride2
	dstcr	0x28, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0xa00, els.extstride2
	dstcr	0x28, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
	dstcr	0x1, elsstatus
	dcp	flowid, r5
	dcp	flowid, r5
	dcp	p30, r7
	dstcr	0, r29
	dshlb	r7, 1, r7
	dstcr	1536, r30
	dandb	r7, 254, r7
	dstcr	2560, r2
	dstcr	29184, r8
	dorb	r7, r9, r7
	dcp	r7, p30
	//APP
	nop <> __iss__ profile stop -msg "copy filtered boxes and scores stop"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "transposestart"
	//NO_APP
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x1, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x0, pls.maskh, west
	dstcr	0xffff0, pls.maskl, west
	dstcr	0x60, pls.stride1, west
	dstcr	0x10, pls.count1, west
	stcr	0x1, bitwidthmode
	dstcr	0x1, pls.count2, west
	dstcr	0x0, plsthresholdwest
	dstcr	0x10, pls.stride2, west
.LBB0_108:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_109 Depth 2
                                        //       Child Loop BB0_110 Depth 3
                                        //       Child Loop BB0_112 Depth 3
                                        //       Child Loop BB0_114 Depth 3
                                        //       Child Loop BB0_116 Depth 3
	dmuli32	r29, r30, r7
	dshlb	r29, 4, r9
	dstcr	0, r18
.LBB0_109:                              //   Parent Loop BB0_108 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_110 Depth 3
                                        //       Child Loop BB0_112 Depth 3
                                        //       Child Loop BB0_114 Depth 3
                                        //       Child Loop BB0_116 Depth 3
	dstcr	0, r19
	dmuli32	r18, r2, r20
	dstcr	0x9, pls.mode, south
	dcpc	r19, cr11
	daddi32	r20, r9, r19
	dstcr	0x300, pc.mode, south
	dshlb	r19, 1, r19
	dstcr	0x200, pc.mode, north
	daddi32	r19, r2, r19
	nrb	cr11, north
	dcp	r19, pls.addr, south
	dstcr	0xa0, pls.stride1, south
	dstcr	0x10, pls.count1, south
	dstcr	0x10, pls.stride2, south
	dcp	r28, dependencyid
	dstcr	0x100, plsstatus, south
	dcp	flowid, r19
	djmpincsetup	0, 16, :.LBB0_110
	dstcr	0x300, pc.mode, south
.LBB0_110:                              //   Parent Loop BB0_108 Depth=1
                                        //     Parent Loop BB0_109 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.111:                             //   in Loop: Header=BB0_109 Depth=2
	dcp	jmpcount, r19
	djmpincsetup	0, 4, :.LBB0_112
	dstcr	0x200, pc.mode, south
.LBB0_112:                              //   Parent Loop BB0_108 Depth=1
                                        //     Parent Loop BB0_109 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.113:                             //   in Loop: Header=BB0_109 Depth=2
	dshlb	r18, 4, r19
	cp	south.0z, cr11
	daddi32	r19, r7, r19
	dstcr	0x200, pc.mode, south
	dshlb	r19, 1, r19
	dstcr	0x8, pls.mode, west
	daddi32	r19, r8, r19
	dstcr	0x1, pc.resetfifo, west
	dstcr	0x200, pc.mode, east
	dstcr	0x300, pc.mode, west
	dcp	r19, pls.addr, west
	dstcr	0x0, dependencyid
	dstcr	0x1000000, plsstatus, west
	dcp	flowid, r19
	djmpincsetup	0, 4, :.LBB0_114
	nrb	cr11, west
	dstcr	0x200, pc.mode, west
.LBB0_114:                              //   Parent Loop BB0_108 Depth=1
                                        //     Parent Loop BB0_109 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	east, west
// %bb.115:                             //   in Loop: Header=BB0_109 Depth=2
	djmpincsetup	0, 16, :.LBB0_116
	dstcr	0x300, pc.mode, west
.LBB0_116:                              //   Parent Loop BB0_108 Depth=1
                                        //     Parent Loop BB0_109 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	east, west
// %bb.117:                             //   in Loop: Header=BB0_109 Depth=2
	dstcr	0x200, pc.mode, west
	djmpincne	r18, 5, :.LBB0_109
// %bb.118:                             //   in Loop: Header=BB0_108 Depth=1
	djmpincne	r29, 10, :.LBB0_108
// %bb.119:
	//APP
	nop <> __iss__ profile stop -msg "transpose stop"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "sorting start"
	//NO_APP
	dstcr	0x9, pls.mode, south
	dstcr	0x300, pc.mode, south
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr11
	stcr	2092, crp2
	addi32	cr11, col, cr11
	dstcr	0x0, pls.maskh, south
	dstcr	0xffff0, pls.maskl, south
	dstcr	0x7200, pls.addr, south
	dstcr	0x10, pls.stride1, south
	dstcr	0x5, pls.count1, south
	dstcr	0xa0, pls.count2, south
	dstcr	0x0, plsthresholdsouth
	dstcr	0x60, pls.stride2, south
	stcr	0x1, bitwidthmode
	dstcr	0x0, dependencyid
	dstcr	0x100, plsstatus, south
	dstcr	0, r28
	addi32	crp1, crp2, crp2
	dcp	flowid, r7
.LBB0_120:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_121 Depth 2
                                        //     Child Loop BB0_123 Depth 2
                                        //     Child Loop BB0_125 Depth 2
	djmpincsetup	0, 5, :.LBB0_121
	dstcr	0x300, pc.mode, south
.LBB0_121:                              //   Parent Loop BB0_120 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.122:                             //   in Loop: Header=BB0_120 Depth=1
	djmpincsetup	0, 4, :.LBB0_123
	dstcr	0x200, pc.mode, south
.LBB0_123:                              //   Parent Loop BB0_120 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.124:                             //   in Loop: Header=BB0_120 Depth=1
	djmpincsetup	0, 11, :.LBB0_125
	dstcr	0x200, pc.mode, south
.LBB0_125:                              //   Parent Loop BB0_120 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.126:                             //   in Loop: Header=BB0_120 Depth=1
	cmplti32	cr11, 80, cr13
	stcr	0, cr12
	predpush	cr13, :.LBB0_128
// %bb.127:                             //   in Loop: Header=BB0_120 Depth=1
	cp	south.0z, cr12
.LBB0_128:                              //   in Loop: Header=BB0_120 Depth=1
	predpop	
	cp	cr12, [crp2.z+=1]
	djmpincne	r28, 160, :.LBB0_120
// %bb.129:
	stcr	2412, crp2
	djmpincsetup	0, 160, :.LBB0_130
	addi32	crp1, crp2, crp2
	dstcr	0x200, pc.mode, south
.LBB0_130:                              // =>This Inner Loop Header: Depth=1
	dcpc	jmpcount, cr11
	cp.lb	cr11, [crp2.z+=1]
// %bb.131:                             // %NodeBlock
	cmplti32	1, cr10, cr12
	stcr	0, cr11
	predpush	cr12, :.LBB0_187
// %bb.185:                             // %LeafBlock25
	cmpeq32	cr10, 2, cr12
	stcr	1, cr11
	predpush	cr12, :.LBB0_186
// %bb.291:
	stcr	2092, crp2
	addi32	crp1, crp2, crp2
	cp	[crp2.z], cr11
	stcr	2094, crp2
	addi32	crp1, crp2, crp2
	cmplte32	[crp2.z], cr11, cr11
	predpush	cr11, :.LBB0_292
// %bb.295:
	stcr	2414, crp2
	addi32	crp1, crp2, crp2
	stcr	0, [crp2.z]
	stcr	2412, crp2
	addi32	crp1, crp2, crp2
	stcr	1, [crp2.z]
.LBB0_292:                              // %Flow50
	predelse	:.LBB0_294
// %bb.293:
	stcr	2414, crp2
	addi32	crp1, crp2, crp2
	stcr	1, [crp2.z]
	stcr	2412, crp2
	addi32	crp1, crp2, crp2
	stcr	0, [crp2.z]
.LBB0_294:                              // %Flow51
	predpop	
	stcr	0, cr11
.LBB0_186:                              // %Flow53
	predpop	
.LBB0_187:                              // %Flow52
                                        // implicit-def: $x28
	predelse	:.LBB0_189
// %bb.188:                             // %Flow52
	dstcr	0, r28
	cmpneq32	cr10, 1, cr11
	dstcr	1, r28
.LBB0_189:                              // %Flow54
	predpop	
	dcpc	r28, cr12
	predpush	cr11, :.LBB0_190
// %bb.296:                             // %NewDefault
	dstcr	0, r28
	dcpc	r28, cr12
.LBB0_190:                              // %Flow55
	predpop	
	predpush	cr12, :.LBB0_192
// %bb.191:
	stcr	2412, crp2
	addi32	crp1, crp2, crp2
	stcr	0, [crp2.z]
.LBB0_192:
	predpop	
	stcr	2092, crp2
	stcr	2092, crp3
	addi32	crp1, crp2, crp2
	addi32	crp1, crp3, crp3
	orb	crp2, 2, crp2
	cp	[crp3.z], cr11
	stcr	65535, cr12
	djmpincsetup	1, 160, :.LBB0_193
.LBB0_193:                              // =>This Inner Loop Header: Depth=1
	andb	cr11, cr12, cr14
	cp	[crp2.z+=1], cr15
	dcpc	jmpcount, cr13
	cmplt32	cr14, cr15, cr14
	csel	cr15, cr11, cr14
	cmplt32	cr13, cr10, cr13
	csel.lb	cr14, cr11, cr11
// %bb.194:
	stcr	0, cr12
	stcr	65535, cr13
	djmpincsetup	0, 31, :.LBB0_195
.LBB0_195:                              // %.preheader72
                                        // =>This Inner Loop Header: Depth=1
	andb	cr11, cr13, cr11
	cmpneq32	cr11, 0, cr14
	shrlb	cr11, 3, cr11
	addi32.lb	cr12, cr14, cr12
// %bb.196:
	stcr	2412, crp2
	stcr	0x2, bitwidthmode
	addi32	crp1, crp2, crp2
	addi32	crp1, 36, crp5
	cp	crp2, crp3
	stcr	2092, crp2
	andb	cr12, 255, cr11
	addi32	crp1, crp2, crp2
	dstcr	0, r28
	cp	crp2, [crp1 + 4]
	stcr	2784, crp2
	addi32	crp1, 40, crp4          //      
	addi32	crp1, crp2, crp2
	cp	crp3, crp6
	cp	crp2, [crp5]
	stcr	2768, crp2
	stcr	0x1, bitwidthmode
	addi32	crp1, crp2, crp2
	stcr	0, [crp2.z]
	addi32	crp1, 32, crp2
	stcr	0x2, bitwidthmode
	cp	crp4, [crp2]
.LBB0_197:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_199 Depth 2
                                        //     Child Loop BB0_201 Depth 2
                                        //     Child Loop BB0_203 Depth 2
                                        //     Child Loop BB0_207 Depth 2
                                        //     Child Loop BB0_209 Depth 2
                                        //     Child Loop BB0_213 Depth 2
	dcpc	r28, cr12
	cmplt32	cr12, cr11, cr12
	predpush	cr12, :.LBB0_216
// %bb.198:                             //   in Loop: Header=BB0_197 Depth=1
	stcr	2768, crp2
	cp	crp3, [crp1 + 3]
	addi32	crp1, crp2, crp2
	djmpincsetup	0, 4, :.LBB0_199
	cp	crp2, crp3
.LBB0_199:                              // %loadstoreloop2
                                        //   Parent Loop BB0_197 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	stcr.lb	0, [crp3+=1]
// %bb.200:                             // %split1
                                        //   in Loop: Header=BB0_197 Depth=1
	stcr	2748, crp2
	djmpincsetup	0, 4, :.LBB0_201
	addi32	crp1, crp2, crp2
	cp	crp2, crp3
.LBB0_201:                              // %loadstoreloop
                                        //   Parent Loop BB0_197 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	stcr.lb	0, [crp3+=1]
// %bb.202:                             // %split
                                        //   in Loop: Header=BB0_197 Depth=1
	stcr	2092, crp2
	dcmpeq32	r28, 0, r29
	addi32	crp1, crp2, crp2
	dshlb	r28, 24, r30
	cp	crp2, crp6
	dcpc	r29, cr12
	cp	[crp1 + 4], crp2
	cmpneq32	cr12, 0, cr12
	csel	crp4, crp2, crp2
	addi32	crp1, 32, crp3
	dshrab	r30, 24, r29
	cp	crp2, [crp3]
	dmuli32	r29, 3, r29
	dstcr	0, r30
	cp	[crp1 + 3], crp3
	stcr	0x1, bitwidthmode
.LBB0_203:                              //   Parent Loop BB0_197 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dcpc	r30, cr12
	cmplt32	cr12, cr10, cr12
	predpush	cr12, :.LBB0_205
// %bb.204:                             //   in Loop: Header=BB0_203 Depth=2
	dcpc	r29, crp5
	shrlb	[crp6.z], crp5, crp5
	stcr	2768, crp2
	notb	crp5, crp5
	addi32	crp1, crp2, crp2
	andb	crp5, 7, crp5
	cp	crp2, crp7
	shlb	crp5, 1, crp5
	addi32	crp7, crp5, crp5
	addi32	[crp5.z], 1, [crp5.z]
.LBB0_205:                              //   in Loop: Header=BB0_203 Depth=2
	predpop	
	addi32	crp6, 2, crp6
	djmpincne	r30, 160, :.LBB0_203
// %bb.206:                             //   in Loop: Header=BB0_197 Depth=1
	stcr	2748, crp2
	dcmpeq32	r28, 0, r30
	addi32	crp1, crp2, crp2
	stcr	0x2, bitwidthmode
	cp	crp2, crp5
	stcr	2768, crp2
	dcpc	r30, cr13
	addi32	crp1, crp2, crp2
	cmpneq32	cr13, 0, cr13
	cp	crp2, crp7
	addi32	crp1, 36, crp2
	stcr	0, cr12
	cp	[crp2], crp2
	addi32	crp5, 2, crp5
	csel	crp3, crp2, crp6
	djmpincsetup	1, 8, :.LBB0_207
	stcr	0x1, bitwidthmode
.LBB0_207:                              //   Parent Loop BB0_197 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cp	[crp7+=1], cr13
	addi32	cr13, cr12, [crp5.z+=1]
	addi32.lb	cr13, cr12, cr12
// %bb.208:                             //   in Loop: Header=BB0_197 Depth=1
	dcmpeq32	r28, 0, r2
	stcr	0x2, bitwidthmode
	addi32	crp1, 36, crp2
	dcpc	r2, cr12
	cp	[crp1 + 4], crp3
	cmpneq32	cr12, 0, cr13
	csel	crp3, crp4, crp3
	cp	[crp2], crp2
	cp	[crp1 + 3], crp4
	cmpneq32	cr12, 0, cr12
	csel	crp2, crp4, crp2
	addi32	crp1, 36, crp4
	dstcr	0, r30
	cp	crp2, [crp4]
	cp	crp3, crp4
	cp	crp3, [crp1 + 4]
.LBB0_209:                              //   Parent Loop BB0_197 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dcpc	r30, cr12
	cmplt32	cr12, cr10, cr12
	predpush	cr12, :.LBB0_211
// %bb.210:                             //   in Loop: Header=BB0_209 Depth=2
	stcr	0x1, bitwidthmode
	dcpc	r29, crp7
	stcr	2748, crp2
	cp	[crp4.z], crp5
	addi32	crp1, crp2, crp2
	shrlb	crp5, crp7, crp7
	addi32	crp1, 32, crp3
	notb	crp7, crp7
	dcpc	r30, cr12
	andb	crp7, 7, crp7
	shlb	crp7, 1, crp7
	addi32	crp2, crp7, crp2
	cp	[crp2.z], crp7
	addi32	crp7, 1, [crp2.z]
	stcr	0x2, bitwidthmode
	shlb	crp7, 1, crp2
	cp	[crp3], crp3
	addi32	crp3, crp2, crp7
	addi32	crp1, 36, crp3
	cp	[crp3], crp3
	stcr	0x1, bitwidthmode
	addi32	crp3, crp2, crp2
	cp	crp5, [crp7.z]
	cp	cr12, [crp2.z]
.LBB0_211:                              //   in Loop: Header=BB0_209 Depth=2
	predpop	
	addi32	crp4, 2, crp4
	djmpincne	r30, 160, :.LBB0_209
// %bb.212:                             //   in Loop: Header=BB0_197 Depth=1
	addi32	crp1, 36, crp2
	stcr	0x2, bitwidthmode
	dstcr	0, r29
	cp	[crp2], crp4
	stcr	0x1, bitwidthmode
.LBB0_213:                              // %.preheader70
                                        //   Parent Loop BB0_197 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dcpc	r29, cr12
	cmplt32	cr12, cr10, cr12
	predpush	cr12, :.LBB0_215
// %bb.214:                             //   in Loop: Header=BB0_213 Depth=2
	shlb	[crp4.z], 1, crp2
	addi32	crp6, crp2, crp2
	cp	[crp2.z], [crp4.z]
.LBB0_215:                              //   in Loop: Header=BB0_213 Depth=2
	predpop	
	addi32	crp4, 2, crp4
	djmpincne	r29, 160, :.LBB0_213
.LBB0_216:                              // %Flow49
                                        //   in Loop: Header=BB0_197 Depth=1
	predpop	
	addi32	crp1, 32, crp2
	stcr	0x2, bitwidthmode
	cp	crp6, crp3
	cp	[crp2], crp4
	djmpincne	r28, 11, :.LBB0_197
// %bb.217:
	stcr	2412, crp2
	cmpneq32	cr11, 0, cr11
	addi32	crp1, crp2, crp2
	dstcr	0, r28
	cp	crp2, crp4
	addi32	crp1, 36, crp2
	cp	crp4, cr12
	cp	[crp2], crp2
	cmpneq32	crp2, cr12, cr12
	stcr	2092, crp2
	andb	cr11, cr12, cr11
	addi32	crp1, crp2, crp2
	cp	crp2, crp5
	predpush	cr11, :.LBB0_221
.LBB0_218:                              // %.preheader68
                                        // =>This Inner Loop Header: Depth=1
	dcpc	r28, cr11
	cmplt32	cr11, cr10, cr11
	predpush	cr11, :.LBB0_220
// %bb.219:                             //   in Loop: Header=BB0_218 Depth=1
	addi32	crp1, 36, crp2
	cp	[crp2], crp2
	stcr	0x1, bitwidthmode
	cp	[crp2.z], [crp4.z]
	addi32	crp1, 32, crp2
	stcr	0x2, bitwidthmode
	cp	[crp2], crp2
	stcr	0x1, bitwidthmode
	cp	[crp2.z], [crp5.z]
.LBB0_220:                              //   in Loop: Header=BB0_218 Depth=1
	predpop	
	addi32	crp1, 36, crp2
	stcr	0x2, bitwidthmode
	addi32	crp1, 36, crp3
	addi32	crp4, 2, crp4
	cp	[crp2], crp2
	addi32	crp5, 2, crp5
	addi32	crp2, 2, crp2
	cp	crp2, [crp3]
	addi32	crp1, 32, crp2
	addi32	crp1, 32, crp3
	cp	[crp2], crp2
	addi32	crp2, 2, crp2
	cp	crp2, [crp3]
	djmpincne	r28, 160, :.LBB0_218
.LBB0_221:                              // %.loopexit69
	predpop	
	cp	row, cr11
	cp	col, cr10
	shlb	row, 4, cr12
	stcr	2092, crp2
	addi32	cr12, col, cr12
	dstcr	0x8, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x7200, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x5, pls.count1, north
	dstcr	0xa0, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x60, pls.stride2, north
	stcr	0x1, bitwidthmode
	dcp	r7, dependencyid
	dstcr	0x1, plsstatus, north
	dstcr	0, r28
	addi32	crp1, crp2, crp2
	dcp	flowid, r7
.LBB0_222:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_225 Depth 2
                                        //     Child Loop BB0_227 Depth 2
	cmplti32	cr12, 80, cr13
	predpush	cr13, :.LBB0_224
// %bb.223:                             //   in Loop: Header=BB0_222 Depth=1
	nrb	[crp2.z], north
.LBB0_224:                              //   in Loop: Header=BB0_222 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB0_225
	dstcr	0x200, pc.mode, north
.LBB0_225:                              //   Parent Loop BB0_222 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.226:                             //   in Loop: Header=BB0_222 Depth=1
	djmpincsetup	0, 5, :.LBB0_227
	dstcr	0x300, pc.mode, north
.LBB0_227:                              //   Parent Loop BB0_222 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.228:                             //   in Loop: Header=BB0_222 Depth=1
	addi32	crp2, 2, crp2
	djmpincne	r28, 160, :.LBB0_222
// %bb.229:
	shlb	cr11, 4, cr11
	stcr	0, crp2
	dstcr	0x200, pc.mode, north
	djmpeqoff	0, r6, :.LBB0_234
// %bb.230:
	stcr	2092, crp3
	stcr	2412, crp4
	dstcr	65535, r29
	stcr	2092, crp5
	dstcr	0, r28
	addi32	crp1, crp3, crp3
	addi32	crp1, crp4, crp4
	addi32	crp1, crp5, crp5
	dandb	r17, r29, r17
.LBB0_231:                              // %.preheader66
                                        // =>This Inner Loop Header: Depth=1
	cp	[crp3.z], cr12
	dcpc	r17, cr13
	cmplte32	cr13, cr12, cr13
	predpush	cr13, :.LBB0_233
// %bb.232:                             //   in Loop: Header=BB0_231 Depth=1
	stcr	2412, crp7
	shlb	crp2, 1, crp6
	addi32	crp1, crp7, crp7
	addi32	crp2, 1, crp2
	addi32	crp7, crp6, crp7
	addi32	crp5, crp6, crp6
	cp	[crp4.z], [crp7.z]
	cp	cr12, [crp6.z]
.LBB0_233:                              //   in Loop: Header=BB0_231 Depth=1
	predpop	
	addi32	crp3, 2, crp3
	addi32	crp4, 2, crp4
	djmpincne	r28, r6, :.LBB0_231
.LBB0_234:                              // %.loopexit67
	addi32	cr11, cr10, cr10
	dstcr	0, r17
	nop <> __iss__ print	crp2 -dec -msg "vcount pruned"
.LBB0_235:                              // =>This Inner Loop Header: Depth=1
	dcpc	r17, cr11
	cmplt32	cr11, crp2, cr11
	dstcr	1, r29
	predpush	cr11, :.LBB0_237
// %bb.236:                             //   in Loop: Header=BB0_235 Depth=1
	dstcr	0, r29
.LBB0_237:                              //   in Loop: Header=BB0_235 Depth=1
	predpop	
	dstcr	1, r6
	dstcr	1, r28
	djmpneqoff	0, r29, :.LBB0_238
// %bb.239:                             // %Landing
                                        //   in Loop: Header=BB0_235 Depth=1
	djmpneqoff	0, r28, :.LBB0_240
.LBB0_241:                              // %Flow45
                                        //   in Loop: Header=BB0_235 Depth=1
	djmpeqoff	r6, 0, :.LBB0_235
	djmp	:.LBB0_242
.LBB0_238:                              // %Break
                                        //   in Loop: Header=BB0_235 Depth=1
	dstcr	0, r28
	djmpeqoff	r28, 0, :.LBB0_241
.LBB0_240:                              //   in Loop: Header=BB0_235 Depth=1
	daddi32	r17, 1, r17
	dcmplt32	159, r17, r6
	djmpeqoff	r6, 0, :.LBB0_235
.LBB0_242:
	dstcr	0, rp2
	cmplt32	cr10, 80, cr11
	addi32	crp1, 36, crp4
	dcpc	rp2, crp3
	nop <> __iss__ print	r17 -dec -msg "vmaxcount"
	//APP
	nop <> __iss__ profile stop -msg "sorting stop"
	//NO_APP
	//APP
	nop <> __iss__ profile start -msg "nms start"
	//NO_APP
	nop <> __iss__ print	r16 -fx12 -msg "overlapthreshFP"
                                        // implicit-def: $cx10
	stcr	0x2, bitwidthmode
	cp	crp3, [crp4]
	predpush	cr11, :.LBB0_246
// %bb.243:
	stcr	0, crp3
	addi32	crp1, 36, crp4
	cmplti32	0, crp2, cr11
	cp	crp3, [crp4]
	stcr	0, crp4
	cp	crp4, cr10
	predpush	cr11, :.LBB0_245
// %bb.244:
	stcr	1, crp3
	addi32	crp1, 36, crp4
	cp	crp3, [crp4]
.LBB0_245:                              // %Flow44
	predpop	
.LBB0_246:
	predpop	
	stcr	2412, crp3
	dstcr	0x0, pls.addr, north
	addi32	crp1, crp3, crp3
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
	stcr	0x1, bitwidthmode
	dstcr	0, r6
	addi32	crp1, 40, crp4          //      
	dstcr	33686018, r28
	cp	[crp3.z], cr11
	stcr	0x2, bitwidthmode
	dstcr	0x0, plsthresholdnorth
.LBB0_247:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_250 Depth 2
                                        //     Child Loop BB0_252 Depth 2
                                        //     Child Loop BB0_254 Depth 2
                                        //     Child Loop BB0_256 Depth 2
                                        //     Child Loop BB0_258 Depth 2
	cmplti32	0, crp2, cr12
	predpush	cr12, :.LBB0_261
// %bb.248:                             //   in Loop: Header=BB0_247 Depth=1
	dmuli32	r6, 160, r29
	cmplt32	cr11, 160, cr14
	stcr	0, cr13
	dcpc	r29, cr12
	addi32	cr12, cr11, cr12
	predpush	cr14, :.LBB0_260
// %bb.249:                             //   in Loop: Header=BB0_247 Depth=1
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	r5, dependencyid
.LBB0_250:                              //   Parent Loop BB0_247 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dandb	r28, plsstatus, r29
	dorb	r29, pelsr, r29
	djmpneqoff	r29, 0, :.LBB0_250
// %bb.251:                             //   in Loop: Header=BB0_247 Depth=1
	shlb	cr12, 2, cr12
	djmpincsetup	0, 4, :.LBB0_252
	dstcr	0x1, plsstatus, north
	nrb	cr12, north
	dstcr	0x260, pc.mode, north
.LBB0_252:                              //   Parent Loop BB0_247 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.253:                             //   in Loop: Header=BB0_247 Depth=1
	djmpincsetup	0, 16, :.LBB0_254
	dstcr	0x360, pc.mode, north
.LBB0_254:                              //   Parent Loop BB0_247 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.255:                             //   in Loop: Header=BB0_247 Depth=1
	djmpincsetup	0, 16, :.LBB0_256
.LBB0_256:                              // %.preheader65
                                        //   Parent Loop BB0_247 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.257:                             //   in Loop: Header=BB0_247 Depth=1
	djmpincsetup	0, 4, :.LBB0_258
	dstcr	0x260, pc.mode, north
.LBB0_258:                              //   Parent Loop BB0_247 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	north, south
// %bb.259:                             //   in Loop: Header=BB0_247 Depth=1
	cp	north, cr13
.LBB0_260:                              // %Flow42
                                        //   in Loop: Header=BB0_247 Depth=1
	predpop	
	cp	cr13, [crp4]
.LBB0_261:                              // %Flow43
                                        //   in Loop: Header=BB0_247 Depth=1
	predpop	
	addi32	crp4, 4, crp4
	djmpincne	r6, 4, :.LBB0_247
// %bb.262:
	dstcr	0, r28
	dstcr	0, r6
	dstcr	364, r31
	djmpeqoff	r17, 0, r31
.LBB0_263:                              // %.preheader63.preheader
	addi32	crp1, 40, crp4          //      
	dstcr	0, r29
	addi32	crp4, 16, crp4
	dstcr	33686018, r30
	addi32	crp4, 4, crp3
	dstcr	0, r28
	cp	crp3, [crp1 + 4]
	stcr	2412, crp3
	dstcr	0, r6
	addi32	crp1, crp3, crp3
	cp	crp3, [crp1 + 3]
	stcr	2092, crp3
	addi32	crp1, crp3, crp3
	cp	crp3, [crp1 + 2]
	dstcr	0x0, plsthresholdnorth
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x10, pls.stride1, north
.LBB0_264:                              // %.preheader63
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_266 Depth 2
                                        //       Child Loop BB0_268 Depth 3
                                        //         Child Loop BB0_270 Depth 4
                                        //         Child Loop BB0_272 Depth 4
                                        //         Child Loop BB0_274 Depth 4
                                        //         Child Loop BB0_276 Depth 4
                                        //         Child Loop BB0_278 Depth 4
                                        //       Child Loop BB0_349 Depth 3
                                        //     Child Loop BB0_301 Depth 2
                                        //       Child Loop BB0_304 Depth 3
                                        //       Child Loop BB0_306 Depth 3
                                        //     Child Loop BB0_311 Depth 2
                                        //     Child Loop BB0_313 Depth 2
                                        //     Child Loop BB0_315 Depth 2
                                        //       Child Loop BB0_318 Depth 3
                                        //       Child Loop BB0_320 Depth 3
	stcr	2784, crp3
	subi32	crp2, cr10, cr11
	addi32	crp1, 40, crp2
	addi32	crp1, crp3, crp3
	daddi32	r29, 1, r29
	cp	[crp2], [crp3]
	stcr	2748, crp3
	addi32	crp1, 44, crp2
	addi32	crp1, crp3, crp3
	cp	[crp2], [crp3]
	stcr	2768, crp3
	addi32	crp1, 48, crp2
	addi32	crp1, crp3, crp3
	cp	[crp2], [crp3]
	stcr	2088, crp3
	addi32	crp1, 52, crp2
	addi32	crp1, crp3, crp3
	cp	[crp2], [crp3]
	stcr	0, crp2
	cp	crp2, cr10
	djmplte	r17, r29, :.LBB0_300
// %bb.265:                             //   in Loop: Header=BB0_264 Depth=1
	stcr	0, crp2
	dshlb	r29, 1, r8
	cp	crp2, cr12
	addi32	crp1, 36, crp2
	cmpneq32	cr12, 0, cr12
	cp	[crp2], crp5
	cp	[crp1 + 3], crp3
	dcpc	r8, crp2
	addi32	crp3, crp2, crp6
	cp	[crp1 + 2], crp3
	dcp	r29, r2
	addi32	crp3, crp2, crp2
	dstcr	0x10, pls.count1, north
	dstcr	0x1, pls.count2, north
.LBB0_266:                              //   Parent Loop BB0_264 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_268 Depth 3
                                        //         Child Loop BB0_270 Depth 4
                                        //         Child Loop BB0_272 Depth 4
                                        //         Child Loop BB0_274 Depth 4
                                        //         Child Loop BB0_276 Depth 4
                                        //         Child Loop BB0_278 Depth 4
                                        //       Child Loop BB0_349 Depth 3
	dcpc	r2, cr13
	cmplt32	cr13, cr11, cr13
	xorb	cr12, 1, cr14
	predpush	cr13, :.LBB0_299
// %bb.267:                             //   in Loop: Header=BB0_266 Depth=2
	stcr	2732, crp3
	dstcr	0x0, pls.addr, north
	addi32	crp1, crp3, crp3
	dstcr	0x1, pc.resetfifo, north
	stcr	0x1, bitwidthmode
	dstcr	0, r8
	cp	crp3, crp4
	cp	[crp6.z], cr13
	stcr	0x2, bitwidthmode
.LBB0_268:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_266 Depth=2
                                        // =>    This Loop Header: Depth=3
                                        //         Child Loop BB0_270 Depth 4
                                        //         Child Loop BB0_272 Depth 4
                                        //         Child Loop BB0_274 Depth 4
                                        //         Child Loop BB0_276 Depth 4
                                        //         Child Loop BB0_278 Depth 4
	dmuli32	r8, 160, r9
	cmplt32	cr13, 160, cr17
	stcr	0, cr16
	dcpc	r9, cr15
	addi32	cr15, cr13, cr15
	predpush	cr17, :.LBB0_280
// %bb.269:                             //   in Loop: Header=BB0_268 Depth=3
	dstcr	0x15, pls.mode, north
	dstcr	0x260, pc.mode, south
	dstcr	0x360, pc.mode, north
	dcp	r5, dependencyid
.LBB0_270:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_266 Depth=2
                                        //       Parent Loop BB0_268 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	dandb	r30, plsstatus, r9
	dorb	r9, pelsr, r9
	djmpneqoff	r9, 0, :.LBB0_270
// %bb.271:                             //   in Loop: Header=BB0_268 Depth=3
	shlb	cr15, 2, cr15
	djmpincsetup	0, 4, :.LBB0_272
	dstcr	0x1, plsstatus, north
	nrb	cr15, north
	dstcr	0x260, pc.mode, north
.LBB0_272:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_266 Depth=2
                                        //       Parent Loop BB0_268 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.273:                             //   in Loop: Header=BB0_268 Depth=3
	djmpincsetup	0, 16, :.LBB0_274
	dstcr	0x360, pc.mode, north
.LBB0_274:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_266 Depth=2
                                        //       Parent Loop BB0_268 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	south, north
// %bb.275:                             //   in Loop: Header=BB0_268 Depth=3
	djmpincsetup	0, 16, :.LBB0_276
.LBB0_276:                              // %.preheader
                                        //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_266 Depth=2
                                        //       Parent Loop BB0_268 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	north, south
// %bb.277:                             //   in Loop: Header=BB0_268 Depth=3
	djmpincsetup	0, 4, :.LBB0_278
	dstcr	0x260, pc.mode, north
.LBB0_278:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_266 Depth=2
                                        //       Parent Loop BB0_268 Depth=3
                                        // =>      This Inner Loop Header: Depth=4
	nnb.lb	north, south
// %bb.279:                             //   in Loop: Header=BB0_268 Depth=3
	cp	north, cr16
.LBB0_280:                              // %Flow36
                                        //   in Loop: Header=BB0_268 Depth=3
	predpop	
	cp	cr16, [crp4+=1]
	djmpincne	r8, 4, :.LBB0_268
// %bb.281:                             //   in Loop: Header=BB0_266 Depth=2
	stcr	2768, crp3
	addi32	crp1, crp3, crp3
	cp	[crp3], cr16
	stcr	2088, crp3
	addi32	crp1, crp3, crp3
	cp	[crp3], cr17
	stcr	2784, crp3
	subi32	cr17, cr16, cr15
	addi32	crp1, crp3, crp3
	cp	[crp3], cr7
	stcr	2748, crp3
	addi32	crp1, crp3, crp3
	cp	[crp3], cr28
	stcr	2740, crp3
	subi32	cr28, cr7, cr29
	addi32	crp1, crp3, crp3
	muli32lohi{12}	cr29, cr15, cr31
	cp	[crp3], cr5
	stcr	2744, crp3
	addi32	crp1, crp3, crp3
	cp	[crp3], cr6
	stcr	2732, crp3
	subi32	cr6, cr5, cr30
	addi32	crp1, crp3, crp3
	cp	[crp3], cr15
	stcr	2736, crp3
	cmplti32	cr15, cr28, cr8
	addi32	crp1, crp3, crp3
	cp	[crp3], cr29
	subi32	cr29, cr15, cr9
	cmplti32	cr7, cr29, cr18
	muli32lohi{12}	cr9, cr30, cr19
	andb	cr18, cr8, cr30
	stcr	0, cr9
	addi32	cr19, cr31, cr31
	predpush	cr30, :.LBB0_283
// %bb.282:                             //   in Loop: Header=BB0_266 Depth=2
	stcr	2748, crp3
	cmplti32	cr28, cr29, cr8
	addi32	crp1, crp3, crp3
	cp	crp3, crp4
	stcr	2732, crp3
	addi32	crp1, crp3, crp3
	cp	crp3, crp7
	cp	[crp1 + 7], crp3
	csel	crp4, crp3, crp4
	stcr	2784, crp3
	cmplti32	cr15, cr7, cr8
	addi32	crp1, crp3, crp3
	csel	crp3, crp7, crp3
	cp	[crp3], cr8
	subi32	[crp4], cr8, cr9
.LBB0_283:                              //   in Loop: Header=BB0_266 Depth=2
	predpop	
	cmplti32	cr5, cr17, cr8
	cmplti32	cr16, cr6, cr19
	stcr	0, cr18
	andb	cr19, cr8, cr8
	predpush	cr8, :.LBB0_285
// %bb.284:                             //   in Loop: Header=BB0_266 Depth=2
	stcr	2088, crp3
	stcr	2768, crp4
	addi32	crp1, crp3, crp3
	cp	[crp1 + 5], crp7
	cmplti32	cr17, cr6, cr18
	addi32	crp1, crp4, crp4
	csel	crp3, crp7, crp3
	cp	[crp1 + 6], crp7
	cmplti32	cr5, cr16, cr18
	csel	crp4, crp7, crp4
	cp	[crp4], cr18
	subi32	[crp3], cr18, cr18
.LBB0_285:                              //   in Loop: Header=BB0_266 Depth=2
	predpop	
	muli32lohi{12}	cr9, cr18, cr18
	stcr	0, cr9
	subi32	cr31, cr18, cr31
	predpush	cr30, :.LBB0_287
// %bb.286:                             //   in Loop: Header=BB0_266 Depth=2
	stcr	2748, crp3
	cp	[crp1 + 7], crp7
	addi32	crp1, crp3, crp3
	cmplti32	cr28, cr29, cr28
	stcr	2732, crp4
	csel	crp3, crp7, crp3
	stcr	2784, crp7
	addi32	crp1, crp4, crp4
	addi32	crp1, crp7, crp7
	cmplti32	cr15, cr7, cr7
	csel	crp7, crp4, crp4
	cp	[crp4], cr7
	subi32	[crp3], cr7, cr9
.LBB0_287:                              //   in Loop: Header=BB0_266 Depth=2
	predpop	
	stcr	0, cr7
	predpush	cr8, :.LBB0_289
// %bb.288:                             //   in Loop: Header=BB0_266 Depth=2
	stcr	2088, crp3
	stcr	2768, crp4
	addi32	crp1, crp3, crp3
	cp	[crp1 + 5], crp7
	cmplti32	cr17, cr6, cr17
	addi32	crp1, crp4, crp4
	csel	crp3, crp7, crp3
	cp	[crp1 + 6], crp7
	cmplti32	cr5, cr16, cr16
	csel	crp4, crp7, crp4
	cp	[crp4], cr16
	subi32	[crp3], cr16, cr7
.LBB0_289:                              //   in Loop: Header=BB0_266 Depth=2
	predpop	
	muli32lohi{12}	cr9, cr7, cr17
	dcpc	r16, cr5
                                        // implicit-def: $cx16
	divi32{12}	cr17, cr31, cr17
	cmplte32	cr5, cr17, cr17
	predpush	cr17, :.LBB0_297
// %bb.290:                             //   in Loop: Header=BB0_266 Depth=2
	addi32	cr10, 1, cr16
.LBB0_297:                              // %Flow34
                                        //   in Loop: Header=BB0_266 Depth=2
	predelse	:.LBB0_298
// %bb.345:                             //   in Loop: Header=BB0_266 Depth=2
	andb	cr14, 1, cr14
	stcr	1, cr12
	predpush	cr14, :.LBB0_347
// %bb.346:                             //   in Loop: Header=BB0_266 Depth=2
	stcr	1, crp3
	stcr	0x1, bitwidthmode
	cp	crp3, cr12
	addi32	crp1, 36, crp3
	cp	[crp6.z], cr13
	stcr	0x2, bitwidthmode
	cmpneq32	cr12, 0, cr12
	addi32	crp1, 36, crp4
	cp	[crp3], crp3
	addi32	crp3, 1, crp3
	cp	crp3, [crp4]
.LBB0_347:                              //   in Loop: Header=BB0_266 Depth=2
	predpop	
	stcr	2412, crp4
	shlb	crp5, 1, crp3
	addi32	crp1, crp4, crp4
	addi32	crp1, 36, crp7
	addi32	crp4, crp3, crp4
	cp	[crp7], crp7
	stcr	0x1, bitwidthmode
	addi32	crp7, 127, cr14
	cp	cr13, [crp4.z]
	stcr	2092, crp4
	cmplt32	crp5, cr14, cr14
	addi32	crp1, crp4, crp4
	cp	crp4, crp7
	addi32	crp5, 1, crp4
	addi32	crp7, crp3, crp3
	cp	[crp2.z], [crp3.z]
	predpush	cr14, :.LBB0_350
// %bb.348:                             // %.loopexit.loopexit
                                        //   in Loop: Header=BB0_266 Depth=2
	addi32	crp1, 32, crp3
	stcr	0x2, bitwidthmode
	djmpincsetup	0, 3, :.LBB0_349
	cp	crp4, [crp3]
	addi32	crp1, 36, crp3
	cp	[crp3], crp4
	subi32	crp5, crp4, crp3
	addi32	crp1, 32, crp5
	shlb	crp3, 4, crp3
	cp	[crp5], crp5
	subi32	crp5, crp4, crp7
	cp	[crp1 + 4], crp4
	addi32	crp4, crp3, crp5
	shlb	crp7, 4, crp3
	addi32	crp1, 40, crp4          //      
	cp	[crp1 + 7], crp7
	addi32	crp4, crp3, crp3
	addi32	crp1, 32, crp4
	cp	[crp4], crp4
	cp	cr15, [crp3]
.LBB0_349:                              // %load-store-loop
                                        //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_266 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	cp.lb	[crp7+=1], [crp5.z+=1]
.LBB0_350:                              // %Flow33
                                        //   in Loop: Header=BB0_266 Depth=2
	predpop	
	cp	crp4, crp5
	cp	cr10, cr16
.LBB0_298:                              // %Flow35
                                        //   in Loop: Header=BB0_266 Depth=2
	predpop	
	cp	cr16, cr10
.LBB0_299:                              // %Flow37
                                        //   in Loop: Header=BB0_266 Depth=2
	predpop	
	addi32	crp6, 2, crp6
	addi32	crp2, 2, crp2
	djmpincne	r2, r17, :.LBB0_266
.LBB0_300:                              // %.loopexit62
                                        //   in Loop: Header=BB0_264 Depth=1
	shlb	row, 4, cr12
	stcr	2092, crp2
	addi32	cr12, col, cr12
	dstcr	0x8, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x7200, pls.addr, north
	dstcr	0x5, pls.count1, north
	dstcr	0xa0, pls.count2, north
	dstcr	0x60, pls.stride2, north
	stcr	0x1, bitwidthmode
	dcp	r7, dependencyid
	dstcr	0x1, plsstatus, north
	dstcr	0, r2
	addi32	crp1, crp2, crp2
	dcp	flowid, r7
.LBB0_301:                              //   Parent Loop BB0_264 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_304 Depth 3
                                        //       Child Loop BB0_306 Depth 3
	cmplti32	cr12, 80, cr13
	predpush	cr13, :.LBB0_303
// %bb.302:                             //   in Loop: Header=BB0_301 Depth=2
	nrb	[crp2.z], north
.LBB0_303:                              //   in Loop: Header=BB0_301 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB0_304
	dstcr	0x200, pc.mode, north
.LBB0_304:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_301 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.305:                             //   in Loop: Header=BB0_301 Depth=2
	djmpincsetup	0, 5, :.LBB0_306
	dstcr	0x300, pc.mode, north
.LBB0_306:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_301 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.307:                             //   in Loop: Header=BB0_301 Depth=2
	addi32	crp2, 2, crp2
	djmpincne	r2, 160, :.LBB0_301
// %bb.308:                             //   in Loop: Header=BB0_264 Depth=1
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr12
	addi32	cr12, col, cr12
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0xea00, pls.addr, north
	dstcr	0x5, pls.count1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x50, pls.stride2, north
	dcp	r28, dependencyid
	dstcr	0x1, plsstatus, north
	cmplti32	cr12, 80, cr12
	dcp	flowid, r28
	predpush	cr12, :.LBB0_310
// %bb.309:                             //   in Loop: Header=BB0_264 Depth=1
	addi32	crp1, 36, crp2
	stcr	0x2, bitwidthmode
	cp	[crp2], crp2
	nrb	crp2, north
.LBB0_310:                              //   in Loop: Header=BB0_264 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB0_311
	dstcr	0x200, pc.mode, north
.LBB0_311:                              //   Parent Loop BB0_264 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.312:                             //   in Loop: Header=BB0_264 Depth=1
	djmpincsetup	0, 5, :.LBB0_313
	dstcr	0x300, pc.mode, north
.LBB0_313:                              //   Parent Loop BB0_264 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.314:                             //   in Loop: Header=BB0_264 Depth=1
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr12
	stcr	2412, crp2
	addi32	cr12, col, cr12
	dstcr	0x8, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0xeb40, pls.addr, north
	dstcr	0x5, pls.count1, north
	dstcr	0xa0, pls.count2, north
	dstcr	0x60, pls.stride2, north
	stcr	0x1, bitwidthmode
	dcp	r6, dependencyid
	dstcr	0x1, plsstatus, north
	dstcr	0, r2
	addi32	crp1, crp2, crp2
	dcp	flowid, r6
.LBB0_315:                              //   Parent Loop BB0_264 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_318 Depth 3
                                        //       Child Loop BB0_320 Depth 3
	cmplti32	cr12, 80, cr13
	predpush	cr13, :.LBB0_317
// %bb.316:                             //   in Loop: Header=BB0_315 Depth=2
	nrb	[crp2.z], north
.LBB0_317:                              //   in Loop: Header=BB0_315 Depth=2
	predpop	
	djmpincsetup	0, 4, :.LBB0_318
	dstcr	0x200, pc.mode, north
.LBB0_318:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_315 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.319:                             //   in Loop: Header=BB0_315 Depth=2
	djmpincsetup	0, 5, :.LBB0_320
	dstcr	0x300, pc.mode, north
.LBB0_320:                              //   Parent Loop BB0_264 Depth=1
                                        //     Parent Loop BB0_315 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nnb.lb	south, north
// %bb.321:                             //   in Loop: Header=BB0_315 Depth=2
	addi32	crp2, 2, crp2
	djmpincne	r2, 160, :.LBB0_315
// %bb.322:                             //   in Loop: Header=BB0_264 Depth=1
	addi32	crp1, 36, crp3
	dstcr	0x200, pc.mode, north
	stcr	0x2, bitwidthmode
	cp	cr11, crp2
	cp	[crp3], crp3
	nop <> __iss__ print	crp3 -dec -msg "vboxcount"
	dstcr	-344, r31
	djmpneqoff	r29, r17, r31
.LBB0_323:                              // %.loopexit64
	shlb	row, 4, cr10
	stcr	2092, crp2
	addi32	cr10, col, cr10
	dstcr	0x8, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0x7200, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x5, pls.count1, north
	dstcr	0xa0, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x60, pls.stride2, north
	stcr	0x1, bitwidthmode
	dcp	r7, dependencyid
	dstcr	0x1, plsstatus, north
	dstcr	0, r17
	addi32	crp1, crp2, crp2
	dcp	flowid, r16
.LBB0_324:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_327 Depth 2
                                        //     Child Loop BB0_329 Depth 2
	cmplti32	cr10, 80, cr11
	predpush	cr11, :.LBB0_326
// %bb.325:                             //   in Loop: Header=BB0_324 Depth=1
	nrb	[crp2.z], north
.LBB0_326:                              //   in Loop: Header=BB0_324 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB0_327
	dstcr	0x200, pc.mode, north
.LBB0_327:                              //   Parent Loop BB0_324 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.328:                             //   in Loop: Header=BB0_324 Depth=1
	djmpincsetup	0, 5, :.LBB0_329
	dstcr	0x300, pc.mode, north
.LBB0_329:                              //   Parent Loop BB0_324 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.330:                             //   in Loop: Header=BB0_324 Depth=1
	addi32	crp2, 2, crp2
	djmpincne	r17, 160, :.LBB0_324
// %bb.331:
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr10
	addi32	cr10, col, cr10
	dstcr	0x10, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0xea00, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x5, pls.count1, north
	dstcr	0x1, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x50, pls.stride2, north
	dcp	r28, dependencyid
	dstcr	0x1, plsstatus, north
	cmplti32	cr10, 80, cr10
	dcp	flowid, r17
	predpush	cr10, :.LBB0_333
// %bb.332:
	addi32	crp1, 36, crp2
	stcr	0x2, bitwidthmode
	cp	[crp2], crp2
	nrb	crp2, north
.LBB0_333:
	predpop	
	djmpincsetup	0, 4, :.LBB0_334
	dstcr	0x200, pc.mode, north
.LBB0_334:                              // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.335:
	djmpincsetup	0, 5, :.LBB0_336
	dstcr	0x300, pc.mode, north
.LBB0_336:                              // =>This Inner Loop Header: Depth=1
	nnb.lb	south, north
// %bb.337:
	dstcr	0x200, pc.mode, north
	shlb	row, 4, cr10
	stcr	2412, crp2
	addi32	cr10, col, cr10
	dstcr	0x8, pls.mode, north
	dstcr	0x1, pc.resetfifo, north
	dstcr	0x200, pc.mode, south
	dstcr	0x300, pc.mode, north
	dstcr	0x0, pls.maskh, north
	dstcr	0xffff0, pls.maskl, north
	dstcr	0xeb40, pls.addr, north
	dstcr	0x10, pls.stride1, north
	dstcr	0x5, pls.count1, north
	dstcr	0xa0, pls.count2, north
	dstcr	0x0, plsthresholdnorth
	dstcr	0x60, pls.stride2, north
	stcr	0x1, bitwidthmode
	dcp	r6, dependencyid
	dstcr	0x1, plsstatus, north
	dstcr	0, r7
	addi32	crp1, crp2, crp2
	dcp	flowid, r5
.LBB0_338:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_341 Depth 2
                                        //     Child Loop BB0_343 Depth 2
	cmplti32	cr10, 80, cr11
	predpush	cr11, :.LBB0_340
// %bb.339:                             //   in Loop: Header=BB0_338 Depth=1
	nrb	[crp2.z], north
.LBB0_340:                              //   in Loop: Header=BB0_338 Depth=1
	predpop	
	djmpincsetup	0, 4, :.LBB0_341
	dstcr	0x200, pc.mode, north
.LBB0_341:                              //   Parent Loop BB0_338 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.342:                             //   in Loop: Header=BB0_338 Depth=1
	djmpincsetup	0, 5, :.LBB0_343
	dstcr	0x300, pc.mode, north
.LBB0_343:                              //   Parent Loop BB0_338 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nnb.lb	south, north
// %bb.344:                             //   in Loop: Header=BB0_338 Depth=1
	addi32	crp2, 2, crp2
	djmpincne	r7, 160, :.LBB0_338
// %bb.172:
	dshlb	r17, 16, r6
	addi32	crp1, 36, crp2
	dshrab	r6, 31, r6
	dstcr	0x200, pc.mode, north
	stcr	0x2, bitwidthmode
	dstcr	32768, r7
	dandb	r6, r17, r28
	cp	[crp2], crp2
	dandb	r17, r7, r7
	dstcr	3, r17
	nop <> __iss__ print	crp2 -dec -msg "vboxcount"
	//APP
	nop <> __iss__ profile stop -msg "nms stop"
	//NO_APP
	dcp	r28, dependencyid
	dstcr	0x2, mode
	dshrlb	r7, 15, r6
	dcp	p30, r7
	dandb	r7, 255, r7
.LBB0_173:                              // =>This Inner Loop Header: Depth=1
	dstcr	1, r28
	dcp	pelsr, r29
	djmpeqoff	r29, 0, :.LBB0_175
// %bb.174:                             //   in Loop: Header=BB0_173 Depth=1
	daddi32	r29, -2, r28
	dcmpeq32	r29, 1, r29
	dshlb	r17, r28, r28
	dcsel	1, r28, r28
	dandb	r7, r28, r28
	dcmpeq32	r28, 0, r28
.LBB0_175:                              // %Flow31
                                        //   in Loop: Header=BB0_173 Depth=1
	djmpeqoff	r28, 0, :.LBB0_173
// %bb.176:
	dstcr	0xea00, els.intaddr
	dcp	r15, els.extaddrl
	dcp	r14, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x140, els.intstride2
	dstcr	0x5, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x140, els.extstride2
	dstcr	0x5, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
	dstcr	0x1, elsstatus
	dcp	flowid, r14
	dcp	flowid, r14
	dcp	p30, r14
	dshlb	r5, 16, r17
	dshlb	r14, 1, r14
	dshrab	r17, 31, r17
	dandb	r14, 254, r14
	dandb	r17, r5, r28
	dorb	r14, r6, r14
	dstcr	32768, r7
	dcp	r14, p30
	dcp	r28, dependencyid
	dandb	r5, r7, r7
	dcp	p30, r14
	dshrlb	r7, 15, r17
	dstcr	3, r5
	dandb	r14, 255, r14
.LBB0_177:                              // =>This Inner Loop Header: Depth=1
	dstcr	1, r15
	dcp	pelsr, r6
	djmpeqoff	r6, 0, :.LBB0_179
// %bb.178:                             //   in Loop: Header=BB0_177 Depth=1
	daddi32	r6, -2, r15
	dcmpeq32	r6, 1, r6
	dshlb	r5, r15, r15
	dcsel	1, r15, r15
	dandb	r14, r15, r15
	dcmpeq32	r15, 0, r15
.LBB0_179:                              // %Flow30
                                        //   in Loop: Header=BB0_177 Depth=1
	djmpeqoff	r15, 0, :.LBB0_177
// %bb.180:
	dstcr	0xeb40, els.intaddr
	dcp	r13, els.extaddrl
	dcp	r12, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x7800, els.intstride2
	dstcr	0x1e0, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x7800, els.extstride2
	dstcr	0x1e0, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
	dstcr	0x1, elsstatus
	dcp	flowid, r12
	dcp	flowid, r12
	dcp	p30, r12
	dshlb	r16, 16, r14
	dshlb	r12, 1, r12
	dstcr	32768, r15
	dshrab	r14, 31, r14
	dandb	r12, 254, r12
	dandb	r16, r15, r15
	dandb	r14, r16, r16
	dorb	r12, r17, r12
	dshrlb	r15, 15, r14
	dcp	r12, p30
	dcp	r16, dependencyid
	dcp	p30, r12
	dstcr	3, r15
	dandb	r12, 255, r12
.LBB0_181:                              // =>This Inner Loop Header: Depth=1
	dstcr	1, r13
	dcp	pelsr, r16
	djmpeqoff	r16, 0, :.LBB0_183
// %bb.182:                             //   in Loop: Header=BB0_181 Depth=1
	daddi32	r16, -2, r13
	dcmpeq32	r16, 1, r16
	dshlb	r15, r13, r13
	dcsel	1, r13, r13
	dandb	r12, r13, r13
	dcmpeq32	r13, 0, r13
.LBB0_183:                              // %Flow
                                        //   in Loop: Header=BB0_181 Depth=1
	djmpeqoff	r13, 0, :.LBB0_181
// %bb.184:
	dstcr	0x7200, els.intaddr
	dcp	r11, els.extaddrl
	dcp	r10, els.extaddrh
	dstcr	0x40, els.intstride1
	dstcr	0x7800, els.intstride2
	dstcr	0x1e0, els.intcount1
	dstcr	0x1, els.intcount2
	dstcr	0x40, els.extstride1
	dstcr	0x7800, els.extstride2
	dstcr	0x1e0, els.extcount1
	dstcr	0x1, els.extcount2
	dstcr	0x0, elsthreshold
	dstcr	0x0, els.mode
	dstcr	0x1, elsstatus
	dcp	flowid, r10
	dcp	flowid, r10
	dcp	p30, r10
	dshlb	r10, 1, r10
	dandb	r10, 254, r10
	dorb	r10, r14, r10
	dcp	r10, p30
	dendk	
                                        // -- End function
	.rodata.str1.4
	.p2align	2               // @.str
.L.str:
	.asciz	"confidencethreshFP"

	.p2align	2               // @.str.1
.L.str.1:
	.asciz	"vcount pruned"

	.p2align	2               // @.str.2
.L.str.2:
	.asciz	"vmaxcount"

	.p2align	2               // @.str.3
.L.str.3:
	.asciz	"overlapthreshFP"

	.p2align	2               // @.str.4
.L.str.4:
	.asciz	"vboxcount"


	.note.GNU-stack
