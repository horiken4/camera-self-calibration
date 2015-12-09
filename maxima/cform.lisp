;;;
;;; cform.lisp -- Maxima output formatter for Programming Language C.
;;; Copyright (C) 2007-2011 Tomohide Naniwa
;;; version 1.2: Aug. 8, 2008
;;;    based on precious contribution by D.C. Hauagge
;;; version 1.3: Nov. 22, 2011
;;;    replace ZL-MEMBER to MEMBER for latest version of Maxima

;;; cform.lisp is free software; you can redistribute it
;;; and/or modify it under the terms of the GNU General Public
;;; License as published by the Free Software Foundation; either
;;; version 2, or (at your option) any later version.

;;; cform.lisp is distributed in the hope that it will be
;;; useful, but WITHOUT ANY WARRANTY; without even the implied
;;; warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
;;; See the GNU General Public License for more details.

;;; Based on f90.lisp.  Copyright statements for f90.lisp follow:
;;; Copyright (C) 2004 James F. Amundson

;;; Based on fortra.lisp. Copyright statements for fortra.lisp follow:
;;;  Copyright (c) 1984,1987 by William Schelter,University of Texas
;;;     All rights reserved
;;;  (c) Copyright 1980 Massachusetts Institute of Technology

(in-package "MAXIMA")
(macsyma-module cform)

(DECLARE-TOP (SPECIAL LB RB	        ;Used for communication with MSTRING.
		  $LOADPRINT	;If NIL, no load message gets printed.
		  1//2 -1//2)
	 (*LEXPR C-PRINT $CFORMMX))

(DEFMSPEC $CFORM (L)
 (SETQ L (FEXPRCHECK L))
 (LET ((VALUE (STRMEVAL L)))
      (COND ((MSETQP L) (SETQ VALUE `((MEQUAL) ,(CADR L) ,(MEVAL L)))))
      (COND ((AND (SYMBOLP L) ($MATRIXP VALUE))
	     ($CFORMMX L VALUE))
	    ((AND (NOT (ATOM VALUE)) (EQ (CAAR VALUE) 'MEQUAL)
		  (SYMBOLP (CADR VALUE)) ($MATRIXP (CADDR VALUE)))
	     ($CFORMMX (CADR VALUE) (CADDR VALUE)))
	    (T (C-PRINT VALUE)))))

;; Some aliases for C language may be omittable for Maxima 5.10.x
(setq c-alias
      '(($POW "pow")
        ($EXP "exp")
        (%SQRT "sqrt")
        (%SIN "sin")
	(%COS "cos")
	(%TAN "tan")
	(%ACOS "acos")
	(%ASIN "asin")
	(%ATAN "atan")
	($ATAN2 "atan2")
	(%LOG "log")
	(%SINH "sinh")
	(%COSH "cosh")
	(%TANH "tanh")
	(%ASINH "asinh")
	(%ACOSH "acosh")
	(%ATANH "atanh")
        (MABS "fabs")
	))

(DEFUN C-PRINT (X &OPTIONAL (STREAM #+Maclisp NIL #-Maclisp *standard-output*)
			&AUX #+PDP10 (TERPRI T) #+PDP10 ($LOADPRINT NIL)
		        ;; This is a poor way of saying that array references
  		        ;; are to be printed with parens instead of brackets.
			(LB #\[ ) (RB #\] )
			;; Definition of heading white space.
			(WHS "    "))
  ;; Restructure the expression for displaying.
  (SETQ X (CSCAN X))
  ;; Linearize the expression using MSTRING.  Some global state must be
  ;; modified for MSTRING to generate using C syntax.  This must be
  ;; undone so as not to modifiy the toplevel behavior of MSTRING.
  (UNWIND-PROTECT
    (DEFPROP MEXPT MSIZE-INFIX GRIND)
    (DEFPROP MMINUS 100. LBP)
    (DEFPROP MSETQ (#\:) STRSYM)  
    (mapc (lambda (x) 
	    (putprop (car x) (get (car x) 'REVERSEALIAS) 'KEEP-RA)
	    (putprop (car x) (cadr x) 'REVERSEALIAS)) c-alias)
    (SETQ X (mstring x))
   ;; Make sure this gets done before exiting this frame.
    (DEFPROP MEXPT MSZ-MEXPT GRIND)
    (REMPROP 'MMINUS 'LBP)
    (mapc (lambda (x) 
	    (putprop (car x) (get (car x) 'KEEP-RA) 'REVERSEALIAS)
	    (remprop (car x) 'KEEP-RA)) c-alias)
  )
  
  ;; MSTRING returns a list of characters. Now print them
  (do ((char 0 (1+ char))
       (line ""))
      ((>= char (length x)))
    (setf line (concatenate 'string line (make-sequence 
					  'string 1 
					  :initial-element (nth char x))))
    (if (>= (length line) 80)
	(let ((break_point -1))
	  (mapc #'(lambda (x)
		    (let ((p (search x line :from-end t))) 
		      (if (and p (> p 0))
			  (setf break_point p))))
		'("+" "-" "*" "/"))
;	  (increment break_point)
	  (setf break_point (+ break_point 1))
	  (if (= break_point 0)
	      (progn (princ line stream) (setf line WHS))
	      (progn
		(princ (subseq line 0 break_point) stream)
		(terpri stream)
		(setf line (concatenate 'string WHS
					(subseq line break_point
						(length line))))))))
    (if (and (= char (1- (length x))) (not (equal line WHS)))
	(princ line stream))
    )
  (princ ";" stream)
  (terpri stream)
  '$done)

(DEFUN CSCAN (E)
 (COND ((ATOM E) (cond ((eq e '$%i) '((mprogn) 0.0 1.0))
		       (t E))) ;%I is (0,1)
; Recent C compilers may have prototype declarathions for math functions.
;       ((AND (EQ (CAAR E) 'MEXPT) (EQ (CADR E) '$%E) (numberp (caddr e)))
;	(LIST '($EXP SIMP) (float (CADDR E))))
       ((AND (EQ (CAAR E) 'MEXPT) (EQ (CADR E) '$%E))
	(LIST '($EXP SIMP) (CSCAN (CADDR E))))
       ((AND (EQ (CAAR E) 'MEXPT) (ALIKE1 (CADDR E) 1//2))
	(LIST '(%SQRT SIMP) (CSCAN (CADR E))))
       ((AND (EQ (CAAR E) 'MEXPT) (ALIKE1 (CADDR E) -1//2))
	(LIST '(MQUOTIENT SIMP) 1 (LIST '(%SQRT SIMP) (CSCAN (CADR E)))))
;       ((and (EQ (CAAR E) 'MEXPT) (numberp (caddr E)))
;	(LIST '($POW SIMP) (CSCAN (CADR E)) (float (CADDR E))))
;       ((and (EQ (CAAR E) 'MEXPT) (numberp (cadr E)))
;	(LIST '($POW SIMP) (float (CADR E)) (CSCAN (CADDR E))))
       ((EQ (CAAR E) 'MEXPT)
	(LIST '($POW SIMP) (CSCAN (CADR E)) (CSCAN (CADDR E))))
       ((AND (EQ (CAAR E) 'MTIMES) (RATNUMP (CADR E))
	     (MEMBER (CADADR E) '(1 -1)))
	(COND ((EQUAL (CADADR E) 1) (CSCAN-MTIMES E))
	      (T (LIST '(MMINUS SIMP) (CSCAN-MTIMES E)))))
       ((EQ (CAAR E) 'RAT)
	(LIST '(MQUOTIENT SIMP) (FLOAT (CADR E)) (FLOAT (CADDR E))))
       ((EQ (CAAR E) 'MRAT) (CSCAN (RATDISREP E)))

	;; x[1,2,3] => x[1][2][3]
       ((AND (EQ (CADDAR E) 'ARRAY) (NOT (EQ (CAAR E) 'MQAPPLY)) )
	(if (> (LENGTH  E) 2) 
	    ;; then
	    (LIST '(MQAPPLY SIMP ARRAY) 
		  (CSCAN (APPEND (LIST (LIST (CAAR E) 'SIMP 'ARRAY))
				 (BUTLAST (CDR E))))
		  (CAR (LAST E)))
	  ;; else
	  E
	  )
	)

       ;;  complex numbers to f77 syntax a+b%i ==> (a,b)
       ((and (memq (caar e) '(mtimes mplus))
	     ((lambda (a) 
		      (and (numberp (cadr a))
			   (numberp (caddr a))
			   (not (zerop1 (cadr a)))
			   (list '(mprogn) (caddr a) (cadr a))))
	      (simplify ($bothcoef e '$%i)))))
       (T (CONS (CAR E) (MAPCAR 'CSCAN (CDR E))))))

(DEFUN CSCAN-MTIMES (E)
       (LIST '(MQUOTIENT SIMP)
	     (COND ((NULL (CDDDR E)) (CSCAN (CADDR E)))
		   (T (CONS (CAR E) (MAPCAR 'CSCAN (CDDR E)))))
	     (FLOAT (CADDR (CADR E)))))

;; Takes a name and a matrix and prints a sequence of C assignment
;; statements of the form
;;  NAME[I][J] = <corresponding matrix element>
;;  The indcies I, J will be counted from 1.

(DEFMFUN $CFORMMX (NAME MAT &OPTIONAL (STREAM #-CL NIL #+CL *standard-output*)
			 &AUX ($LOADPRINT NIL) (K 'array))
  (COND ((NOT (symbolp NAME))
	 (MERROR "~%First argument to CFORMMX must be a symbol."))
	((NOT ($MATRIXP MAT))
	 (MERROR "Second argument to CFORMMX not a matrix: ~M" MAT)))
  (DO ((MAT (CDR MAT) (CDR MAT)) (I 1 (f1+ I))) ((NULL MAT))
      (DECLARE (FIXNUM I))
      (DO ((M (CDAR MAT) (CDR M)) (J 1 (f1+ J))) ((NULL M))
	  (DECLARE (FIXNUM J))
	  (C-PRINT `((MEQUAL) ((((,NAME ,K) ,I) ,K) ,J) ,(CAR M)) STREAM)))
  '$DONE)

;; End:
