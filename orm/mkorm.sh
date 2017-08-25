#!/bin/bash
TGT=all
if [ "$1" = "-h" ]; then
	echo "$0 [TARGET|COMPILE_TYPE...]"
	echo "   make TARGET, for various flavors of dstorm"
	echo "   Default TARGET       : make ${TGT}"
	echo "   Default COMPILE_TYPE : DEF ORM MPI GPU"
	echo "   COMPILE_TYPE sets environment variables"
	echo "      {default, WITH_LIBORM, WITH_MPI, WITH_GPU}"
	echo "   changing compilation targets/options/complexity"
else
	COMPILATIONS=
	while [ ! -z "$1" ]; do
		if [ "$1" = "DEF" ]; then COMPILATIONS="$COMPILATIONS DEF"; shift; continue; fi
		if [ "$1" = "ORM" ]; then COMPILATIONS="$COMPILATIONS ORM"; shift; continue; fi
		if [ "$1" = "MPI" ]; then COMPILATIONS="$COMPILATIONS MPI"; shift; continue; fi
		if [ "$1" = "GPU" ]; then COMPILATIONS="$COMPILATIONS GPU"; shift; continue; fi
		TGT="$1"; shift
	done
	if [ -z "$COMPILATIONS" ]; then
		COMPILATIONS="DEF ORM MPI GPU"
	fi
	echo "Goal: make ${TGT}    Compile types: ${COMPILATIONS}"
	OOPS=''
	for COMP in $COMPILATIONS; do
		case $COMP in
			DEF*)
				MKLOG=mk-DEF-${TGT}.log
				echo "Trying ${MKLOG}"
				make realclean >& /dev/null
				make ${TGT} 2>&1 | tee ${MKLOG} && echo YAY || OOPS=${MKLOG}
				;;
			ORM*)
				MKLOG=mk-ORM-${TGT}.log
				echo "Trying ${MKLOG}"
				make realclean >& /dev/null
				WITH_LIBORM=1 make ${TGT} 2>&1 | tee ${MKLOG} && echo YAY || OOPS=${MKLOG}
				;;
			MPI*)
				MKLOG=mk-MPI-${TGT}.log
				echo "Trying ${MKLOG}"
				make realclean >& /dev/null
				WITH_MPI=1 make ${TGT} 2>&1 | tee ${MKLOG} && echo YAY || OOPS=${MKLOG}
				;;
			GPU*)
				make realclean >& /dev/null
				MKLOG=mk-GPU-${TGT}.log
				echo "Trying ${MKLOG}"
				WITH_GPU=1 make ${TGT} 2>&1 | tee ${MKLOG} && echo YAY || OOPS=${MKLOG}
				;;
			*)
				echo "Skipping unrecognized compilation type <$COMP>"
				;;
		esac

		if [ ! -z ${OOPS} ]; then
			break
		fi
		echo "OK   --- ${MKLOG}"
	done
fi
