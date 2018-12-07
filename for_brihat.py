"""
Python implementation of sparse matrix for Laplace heat equation.
"""

import sys
import getopt
import numpy as np
import math
import time

from scipy.sparse import coo_matrix, isspmatrix_coo
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse import linalg

def usage():
   print("Usage for {}", sys.argv[0])
   print("  --help|-h     : write this help")
   print("  -N # |--N=#   : specify number of solution points in 2d")
   print("  -m # |--maxiters=#: maximum number of iterations")
   print("  -w | -write   : save the solution for plotting")

iteration_count = 0

def jacobi_solver_csr( A, b, maxiters = 500, tol = 1e-5 ):
   """
   Python implementation of sparse matrix iterative solver
   using Jacobi's method.
   """

   if not isspmatrix_csr(A):
      print('Invalid sparse format: must be CSR')
      return None, -1, 0

   nrows = b.shape[0]

   # Create the solution vector and initialize the zero.
   x = np.zeros(nrows,dtype='d')

   # Use ||b|| to normalize the residual norm.
   normb = np.linalg.norm(b)

   # Extract the matrix diagonal into a vector.
   D = np.empty_like(x)

   if True:
      err = nrows

      # CSR matrices store the data in compressed form by rows sequentially.
      # The non-zero values and their column indices are stored in the
      # class member numpy arrays A.indices[] and A.data[].
      # The # of non-zero values on row i can be found like this:
      #    nnz = A.indptr[i+1] - A.indptr[i]
      # Here, indptr[] is the pointer offset array and indptr[i] is the index
      # of the first element on row i in the indices[] and data[] arrays.

      # Sweep through the rows (i) and then look for the diagonal column index
      # on each row. Store that value in D[i]. Check that there's a diagonal, too.
      for i in range(nrows):
         indptr = A.indptr[i]
         nnz = A.indptr[i+1] - indptr
         diag = None
         for k in range(indptr,indptr+nnz):
            if i == A.indices[k]:
               diag = A.data[k]
               break

         if diag == None:
            err = i
            break
         else:
            D[i] = diag

      if err < nrows:
         print('No diagonal found for row ' + str(err))
         return None, -2, 0

   # We need a scratch array to hold the intermediate solution each iteration.
   xnew = np.empty_like(x)

   it = 0

   while it < maxiters:
      it += 1

      # We need multiple(A,x) for both the residual test and to compute
      # the next iterate. Since this is the big cost, pre-compute and save.
      Ax = A * x

      res = np.linalg.norm(b - Ax)
      if it % 10 == 0:
         print("iter: %d %e"%(it,res))
      if tol*normb > res:
         break

      # Ax = b -> (L + D + U) x = b
      # Dx + (L+U)x = b -> Dx = b - (L+U)x
      # ...
      # x^(k+1) = D^-1 [ b - (L + U)x ]
      #         = D^-1 [ b - (Ax - Dx) ]

      xnew = ( b - Ax + D * x ) / D
      x = xnew

   return x, ( it < maxiters ), it

def main():

   N = 20
   write_solution = False
   maxiters = 100
   tolerance = 1e-5

   # Load CLI options
   try:
      opts, args = getopt.getopt(sys.argv[1:], "hN:m:wt:", ["help", "N=","maxiters=","write","tolerance"])
   except getopt.GetoptError as err:
      # print help information and exit:
      print(err) # will print something like "option -a not recognized"
      usage()
      sys.exit(2)

   for key, val in opts:
      if key in ("-h", "--help"):
         usage()
         sys.exit()
      elif key in ("-w", "--write"):
         write_solution = True
      elif key in ("-N", "--N"):
         N = int(val)
      elif key in ("-m", "--maxiters"):
         maxiters = int(val)
      elif key in ("-t", "--tolerance"):
         tolerance = double(val)
      else:
         assert False, "Unhandled option: " + key

   print("N         = " + str(N))
   print("Max Iters = " + str( maxiters))
   print("Tolerance = " + str(tolerance))

   # We're going to create a 2d laplacian on a mesh
   # and construct a sparse matrix in COO Format. This
   # requires that we store the matrix value and the
   # row/column of each value into lists. Each non-zero
   # value in the matrix is stored in the following three
   # lists: values[], rowidx[] and colidx[]. The mapping from
   # sparse COO format to a dense matrix format for any non-zero
   # element k is:
   # A[rowidx[k],colidx[k]] = values[k]

   A_values = []
   A_rowidx = []
   A_colidx = []

   nrows = N**2

   b = np.zeros(nrows,dtype='d')

   # Flatten the 2d lattice index into a row-major format.
   def get_index(i,j):
      return i + j * N

   # Define the lattice size, spacing, and the boundary-conditions.
   h = 1.0 / (N+1)
   LeftWall = 1.0
   RightWall = 0.0
   TopWall = 1.0
   BottomWall = 0.0

   # Sweep over all interior lattice points and define the Laplacian
   # 5pt stencil and the right-hand-side (RHS) vector. The known BC values
   # are explicitly factored to the RHS vector.
   # The finite-difference model for Laplacian(u) = 0 is:
   #    (u[i+1,j] - 2 u[i,j] + u[i-1,j]) / h**2 + 
   #    (u[i,j+1] - 2 u[i,j] + u[i,j-1]) / h**2 = 0
   # Wall values are known and moved to the RHS vector.
   # Factor the unknowns in A u = b
   
   for i in range(N):
      for j in range(N):
         # The 5pt Laplacian is
         # 0  1  0
         # 1 -4  1
         # 0  1  0
         
         # i,j are the lattice indexes here. Convert to a flattened matrix row index.
         rowidx = get_index(i,j)

         A_rowidx.append( rowidx )
         A_colidx.append( rowidx )
         A_values.append( -4.0 )

         # Left point
         if i == 0:
            b[rowidx] -= LeftWall * h**2
         else:
            A_values.append( 1.0 )
            A_rowidx.append( rowidx )
            A_colidx.append( get_index(i-1,j) )

         # Right point
         if i == N-1:
            b[rowidx] -= RightWall * h**2
         else:
            A_values.append( 1.0 )
            A_rowidx.append( rowidx )
            A_colidx.append( get_index(i+1,j) )

         # Below point
         if j == 0:
            b[rowidx] -= BottomWall * h**2
         else:
            A_values.append( 1.0 )
            A_rowidx.append( rowidx )
            A_colidx.append( get_index(i,j-1) )

         # Above point
         if j == N-1:
            b[rowidx] -= TopWall * h**2
         else:
            A_values.append( 1.0 )
            A_rowidx.append( rowidx )
            A_colidx.append( get_index(i,j+1) )

   # Now, instantiate a COO matrix.
   A_coo = coo_matrix(( np.array(A_values), (np.array(A_rowidx), np.array(A_colidx))),
                            shape=(N**2, N**2))

   # Finally, convert the COO to a CSR matrix (which is faster).
   A_csr = csr_matrix(A_coo)

   if nrows < 100:
      print(A_csr)
      print(A_csr.todense())

   def write_tofile(x,filename='x.dat'):
      fp = open(filename, 'w')
      for i in range(N):
         for j in range(N):
            ij = get_index(i,j)
            fp.write( str(i*h) + " " + str(j*h) + " " + str(x[ij]) + "\n" )
      fp.close()

      return

   normb = np.linalg.norm(b)
   print("norm(b) = %e"%(normb))

   # Run the CG solver.

   if True:

      # The built-in CG solver allows a user-defined 'callback' function that's
      # called each iteration so the user can inspect how the solution is
      # progressing. I'll just compute and print the 2-norm of the residual every
      # 10th step. iteration_count, b, and A_csr are known externally.
      def cg_callback(xk):
         global iteration_count
         iteration_count += 1
         if iteration_count % 10 == 0:
            print("iter: %d %e"%(iteration_count, np.linalg.norm(b - A_csr * xk)))

      print("Starting CG solver:")

      time_start = time.time()
      x, info = linalg.cg( A_csr, b, tol=tolerance, maxiter=500, callback=cg_callback )

      time_stop = time.time()
      if info == 0:
         print("CG solver converged in %d iterations and took %8.4f (seconds)"%(iteration_count, time_stop-time_start))
         if write_solution:
            write_tofile(x, 'cg.dat')
      else:
         print('CG solver failed to converge. Error flag= ' + str(info))

   # Run the Jacobi Solver
   if True:

      print("Starting Jacobi solver:")

      time_start = time.time()

      # The Jacobi iteration is much simplier than CG. In matrix form,
      # it's ...
      #    x^(k+1) = D^-1 [ b - (L + U)x ]
      #
      # This comes from additively splitting A into (L + D + U) where
      # L is the strictly lower triangular matrix,
      # D is the diagonal,
      # U is the strictly upper triangular matrix.
      # Instead of actually splitting the matrix into L+D+U (i.e., 3 matrices),
      # I'll just create a vector that holds the diagonal terms and use this
      # form ...
      #    x^(k+1) = D^-1 [ b - (Ax - Dx) ]
      #
      # This is easier to code since Ax can be done with 1 command and the
      # inverse of a diagonal matrix is just 1/d_i,j ... so I can just divide
      # of the diagonal element-wise.

      x, converged, niters = jacobi_solver_csr( A_csr, b, maxiters=maxiters, tol=tolerance )

      time_stop = time.time()
      if converged is True:
         print('Jacobi solver converged in %d iterations and took %8.4f (seconds)'%(niters,time_stop-time_start))
         if write_solution:
            write_tofile(x, 'jac.dat')
      else:
         print('Jacobi solver failed to converge in %d iterations and %8.4f (seconds)'%(niters,time_stop-time_start))

if __name__ == "__main__":
   main()
