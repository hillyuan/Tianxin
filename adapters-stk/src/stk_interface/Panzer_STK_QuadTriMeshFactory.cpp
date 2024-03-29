// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#include <Panzer_STK_QuadTriMeshFactory.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <PanzerAdaptersSTK_config.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

namespace panzer_stk {

QuadTriMeshFactory::QuadTriMeshFactory()
{
   initializeWithDefaults();
}

//! Destructor
QuadTriMeshFactory::~QuadTriMeshFactory()
{
}

//! Build the mesh object
Teuchos::RCP<STK_Interface> QuadTriMeshFactory::buildMesh(stk::ParallelMachine parallelMach) const
{
   PANZER_FUNC_TIME_MONITOR("panzer::QuadTriMeshFactory::buildMesh()");

   // build all meta data
   RCP<STK_Interface> mesh = buildUncommitedMesh(parallelMach);

   // commit meta data
   mesh->initialize(parallelMach);

   // build bulk data
   completeMeshConstruction(*mesh,parallelMach);

   return mesh;
}

Teuchos::RCP<STK_Interface> QuadTriMeshFactory::buildUncommitedMesh(stk::ParallelMachine parallelMach) const
{
   PANZER_FUNC_TIME_MONITOR("panzer::QuadTriMeshFactory::buildUncomittedMesh()");

   RCP<STK_Interface> mesh = rcp(new STK_Interface(2));

   machRank_ = stk::parallel_machine_rank(parallelMach);
   machSize_ = stk::parallel_machine_size(parallelMach);

   if (xProcs_ == -1 && yProcs_ == -1) {
     // copied from galeri
     xProcs_ = yProcs_ = Teuchos::as<int>(pow(Teuchos::as<double>(machSize_), 0.5));

     if (xProcs_ * yProcs_ != Teuchos::as<int>(machSize_))  {
       // Simple method to find a set of processor assignments
       xProcs_ = yProcs_ = 1;

       // This means that this works correctly up to about maxFactor^2
       // processors.
       const int maxFactor = 100;

       int ProcTemp = machSize_;
       int factors[maxFactor];
       for (int jj = 0; jj < maxFactor; jj++) factors[jj] = 0;
       for (int jj = 2; jj < maxFactor; jj++) {
         bool flag = true;
         while (flag) {
           int temp = ProcTemp/jj;
           if (temp*jj == ProcTemp) {
             factors[jj]++;
             ProcTemp = temp;

           } else {
             flag = false;
           }
         }
       }
       xProcs_ = ProcTemp;
       for (int jj = maxFactor-1; jj > 0; jj--) {
         while (factors[jj] != 0) {
           if      (xProcs_ <= yProcs_) xProcs_ = xProcs_*jj;
           else                         yProcs_ = yProcs_*jj;
           factors[jj]--;
         }
       }
     }

   } else if(xProcs_==-1) {
      // default x only decomposition
      xProcs_ = machSize_; 
      yProcs_ = 1;
   }
   TEUCHOS_TEST_FOR_EXCEPTION(int(machSize_)!=xProcs_*yProcs_,std::logic_error,
                      "Cannot build QuadTriMeshFactory, the product of \"X Procs\" and \"Y Procs\""
                      " must equal the number of processors.");
   procTuple_ = procRankToProcTuple(machRank_);

   // build meta information: blocks and side set setups
   buildMetaData(parallelMach,*mesh);

   mesh->addPeriodicBCs(periodicBCVec_);
   mesh->setBoundingBoxSearchFlag(useBBoxSearch_);
 
   return mesh;
}

void QuadTriMeshFactory::completeMeshConstruction(STK_Interface & mesh,stk::ParallelMachine parallelMach) const
{
   PANZER_FUNC_TIME_MONITOR("panzer::QuadTriMeshFactory::completeMeshConstruction()");

   if(not mesh.isInitialized())
      mesh.initialize(parallelMach);

   // add node and element information
   buildElements(parallelMach,mesh);

   // finish up the edges
   mesh.buildSubcells();
   mesh.buildLocalElementIDs();
 //  if(createEdgeBlocks_) {
  //    mesh.buildLocalEdgeIDs();
 //  }
   mesh.buildLocalSideIDs();
   
   // now that edges are built, sidets can be added
   addSideSets(mesh);

   // add nodesets
   addNodeSets(mesh);
	
//   mesh.applyPeriodicCondition();
  // mesh.buildLocalElementIDs();
 //  mesh.buildLocalNodeIDs();

   // calls Stk_MeshFactory::rebalance
   this->rebalance(mesh);
}

//! From ParameterListAcceptor
void QuadTriMeshFactory::setParameterList(const Teuchos::RCP<Teuchos::ParameterList> & paramList)
{
   paramList->validateParametersAndSetDefaults(*getValidParameters(),0);

   setMyParamList(paramList);

   x0_ = paramList->get<double>("X0"); 
   y0_ = paramList->get<double>("Y0"); 

   xf_ = paramList->get<double>("Xf"); 
   yf_ = paramList->get<double>("Yf"); 

   xBlocks_ = paramList->get<int>("X Blocks");
   yBlocks_ = paramList->get<int>("Y Blocks");

   nXElems_ = paramList->get<int>("X Elements");
   nYElems_ = paramList->get<int>("Y Elements");

   xProcs_ = paramList->get<int>("X Procs");
   yProcs_ = paramList->get<int>("Y Procs");

   // read in periodic boundary conditions
   parsePeriodicBCList(Teuchos::rcpFromRef(paramList->sublist("Periodic BCs")),periodicBCVec_,useBBoxSearch_);
}

//! From ParameterListAcceptor
Teuchos::RCP<const Teuchos::ParameterList> QuadTriMeshFactory::getValidParameters() const
{
   static RCP<Teuchos::ParameterList> defaultParams;

   /* fill with default values
   
      5----6----7--8--9
	  |    |    | /| /|
	  |    |    |/ |/ |
      0----1----2--3--4
	  
	  element block 0: Quad 0,1
	  element block 1: Tri  2,3,4,5
   */
   if(defaultParams == Teuchos::null) {
      defaultParams = rcp(new Teuchos::ParameterList);

      defaultParams->set<double>("X0",0.0);
      defaultParams->set<double>("Y0",0.0);

      defaultParams->set<double>("Xf",4.0);
      defaultParams->set<double>("Yf",1.0);

      defaultParams->set<int>("X Blocks",2);
      defaultParams->set<int>("Y Blocks",1);

      defaultParams->set<int>("X Procs",-1);
      defaultParams->set<int>("Y Procs",1);

      defaultParams->set<int>("X Elements",4);
      defaultParams->set<int>("Y Elements",1);

      Teuchos::ParameterList & bcs = defaultParams->sublist("Periodic BCs");
      bcs.set<int>("Count",0); // no default periodic boundary conditions
   }

   return defaultParams;
}

void QuadTriMeshFactory::initializeWithDefaults()
{
   // get valid parameters
   RCP<Teuchos::ParameterList> validParams = rcp(new Teuchos::ParameterList(*getValidParameters()));

   // set that parameter list
   setParameterList(validParams);
}

void QuadTriMeshFactory::buildMetaData(stk::ParallelMachine /* parallelMach */, STK_Interface & mesh) const
{
   typedef shards::Triangle<> TriTopo;
   const CellTopologyData * ctd_tri = shards::getCellTopologyData<TriTopo>();
   const CellTopologyData * side_ctd = shards::CellTopology(ctd_tri).getBaseCellTopologyData(1,0);
   typedef shards::Quadrilateral<4> QuadTopo;
   const CellTopologyData * ctd_quad = shards::getCellTopologyData<QuadTopo>();
 //  const CellTopologyData * side_ctd = shards::CellTopology(ctd).getBaseCellTopologyData(1,0);

   // build meta data
   //mesh.setDimension(2);
   for(int bx=0;bx<xBlocks_;bx++) {
      for(int by=0;by<yBlocks_;by++) {

         // add this element block
         {
            std::stringstream ebPostfix;
            ebPostfix << "-" << bx << "_" << by;

            // add element blocks
			if( bx==0 ) 
            	mesh.addElementBlock("eblock"+ebPostfix.str(),ctd_quad);
			else
				mesh.addElementBlock("eblock"+ebPostfix.str(),ctd_tri);
         }

      }
   }

   // add sidesets 
   mesh.addSideset("left",side_ctd);
   mesh.addSideset("right",side_ctd);
   mesh.addSideset("top",side_ctd);
   mesh.addSideset("bottom",side_ctd);

   // add nodesets
   mesh.addNodeset("origin");
}

void QuadTriMeshFactory::buildElements(stk::ParallelMachine parallelMach,STK_Interface & mesh) const
{
   mesh.beginModification();
      // build each block
      for(int xBlock=0;xBlock<xBlocks_;xBlock++) {
         for(int yBlock=0;yBlock<yBlocks_;yBlock++) {
            buildBlock(parallelMach,xBlock,yBlock,mesh);
         }
      }
   mesh.endModification();
}

void QuadTriMeshFactory::buildBlock(stk::ParallelMachine /* parallelMach */, int xBlock, int yBlock, STK_Interface& mesh) const
{
   // grab this processors rank and machine size
  // std::pair<int,int> sizeAndStartX = determineXElemSizeAndStart(xBlock,xProcs_,machRank_);
  // std::pair<int,int> sizeAndStartY = determineYElemSizeAndStart(yBlock,yProcs_,machRank_);

  // int myXElems_start = sizeAndStartX.first;
  // int myXElems_end  = myXElems_start+sizeAndStartX.second;
  // int myYElems_start = sizeAndStartY.first;
  // int myYElems_end  = myYElems_start+sizeAndStartY.second;
   int totalXElems = nXElems_*xBlocks_;
   int totalYElems = nYElems_*yBlocks_;

   double deltaX = (xf_-x0_)/double(totalXElems);
   double deltaY = (yf_-y0_)/double(totalYElems);

   std::vector<double> coord(2,0.0);

   // build the nodes, 1-based
   if( xBlock==0 ) {
     for(int nx=0;nx<3;++nx) {
       coord[0] = this->getMeshCoord(nx, deltaX, x0_);
       for(int ny=0;ny<2;++ny) {
         coord[1] = this->getMeshCoord(ny, deltaY, y0_);
         mesh.addNode(ny*(totalXElems+1)+nx+1,coord);
       }
     }
   } else {
	 for(int nx=3;nx<5;++nx) {
       coord[0] = this->getMeshCoord(nx, deltaX, x0_);
       for(int ny=0;ny<2;++ny) {
         coord[1] = this->getMeshCoord(ny, deltaY, y0_);
         mesh.addNode(ny*(totalXElems+1)+nx+1,coord);
       }
	 }
   }

   std::stringstream blockName;
   blockName << "eblock-" << xBlock << "_" << yBlock;
   stk::mesh::Part * block = mesh.getElementBlockPart(blockName.str());

   
   // build the elements
   if( xBlock==0 ) {  //quad 
	 stk::mesh::EntityId gid = 0 ;
     for(int nx=0;nx<2;++nx) {
       for(int ny=0;ny<1;++ny) {
         ++gid;
         std::vector<stk::mesh::EntityId> nodes(4);
         nodes[0] = nx+1+ny*(totalXElems+1);
         nodes[1] = nodes[0]+1;
         nodes[2] = nodes[1]+(totalXElems+1);
         nodes[3] = nodes[2]-1;

         mesh.addElement(gid,nodes,block);
       }
     }
   } else { //tri
	 stk::mesh::EntityId gid = 2 ;
     for(int nx=2;nx<4;++nx) {
       for(int ny=0;ny<1;++ny) {
         std::vector<stk::mesh::EntityId> nodes(3);
         stk::mesh::EntityId sw,se,ne,nw;
         sw = nx+1+ny*(totalXElems+1);
         se = sw+1;
         ne = se+(totalXElems+1);
         nw = ne-1;

         nodes[0] = sw;
         nodes[1] = se;
         nodes[2] = ne;
		 ++gid;
         mesh.addElement(gid,nodes,block);

         nodes[0] = sw;
         nodes[1] = ne;
         nodes[2] = nw;
		 ++gid;
         mesh.addElement(gid,nodes,block);
       }
     }
   }
}

void QuadTriMeshFactory::addSideSets(STK_Interface & mesh) const
{
   mesh.beginModification();

   std::size_t totalXElems = nXElems_*xBlocks_;
   std::size_t totalYElems = nYElems_*yBlocks_;

   // get all part vectors
   stk::mesh::Part * left = mesh.getSideset("left");
   stk::mesh::Part * right = mesh.getSideset("right");
   stk::mesh::Part * top = mesh.getSideset("top");
   stk::mesh::Part * bottom = mesh.getSideset("bottom");

   std::vector<stk::mesh::Entity> localElmts;
   mesh.getMyElements(localElmts);

   // loop over elements adding edges to sidesets
   std::vector<stk::mesh::Entity>::const_iterator itr;
   for( const auto& element : localElmts ) {
      stk::mesh::EntityId gid = mesh.EntityGlobalId(element);

      bool lower = (gid%2 != 0);
      std::size_t block = lower ? (gid+1)/2 : gid/2;
      std::size_t nx,ny;
      ny = (block-1) / totalXElems;
      nx = block-ny*totalXElems-1;

      // vertical boundaries
      ///////////////////////////////////////////

      if(nx+1==totalXElems && lower) { 
         stk::mesh::Entity edge = mesh.findConnectivityById(element, stk::topology::EDGE_RANK, 1);

         // on the right
         if(mesh.entityOwnerRank(edge)==machRank_)
            mesh.addEntityToSideset(edge,right);
      }

      if(nx==0 && !lower) {
         stk::mesh::Entity edge = mesh.findConnectivityById(element, stk::topology::EDGE_RANK, 2);

         // on the left
         if(mesh.entityOwnerRank(edge)==machRank_)
            mesh.addEntityToSideset(edge,left);
      }

      // horizontal boundaries
      ///////////////////////////////////////////

      if(ny==0 && lower) {
         stk::mesh::Entity edge = mesh.findConnectivityById(element, stk::topology::EDGE_RANK, 0);

         // on the bottom
         if(mesh.entityOwnerRank(edge)==machRank_)
            mesh.addEntityToSideset(edge,bottom);
      }

      if(ny+1==totalYElems && !lower) {
         stk::mesh::Entity edge = mesh.findConnectivityById(element, stk::topology::EDGE_RANK, 1);

         // on the top
         if(mesh.entityOwnerRank(edge)==machRank_)
            mesh.addEntityToSideset(edge,top);
      }
   }

   mesh.endModification();
}

void QuadTriMeshFactory::addNodeSets(STK_Interface & mesh) const
{
   mesh.beginModification();

   // get all part vectors
   stk::mesh::Part * origin = mesh.getNodeset("origin");

   Teuchos::RCP<stk::mesh::BulkData> bulkData = mesh.getBulkData();
   if(machRank_==0) 
   {
      stk::mesh::Entity node = bulkData->get_entity(mesh.getNodeRank(),1);

      // add zero node to origin node set
      mesh.addEntityToNodeset(node,origin);
   }

   mesh.endModification();
}

//! Convert processor rank to a tuple
Teuchos::Tuple<std::size_t,2> QuadTriMeshFactory::procRankToProcTuple(std::size_t procRank) const
{
   std::size_t i=0,j=0;

   j = procRank/xProcs_; 
   procRank = procRank % xProcs_;
   i = procRank;

   return Teuchos::tuple(i,j);
}

} // end panzer_stk
