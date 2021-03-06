// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) Xi Yuan
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
// THIS SOFTWARE IS PROVIDED THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ***********************************************************************
// @HEADER

#ifndef _TIANXIN_ABSTRACTDISCRTATION_HPP
#define _TIANXIN_ABSTRACTDISCRTATION_HPP

#include <string>
#include <vector>

namespace panzer 
{	
class PhysicsBlock;
}

namespace TianXin {

struct AbstractDiscretation {
	virtual unsigned getDimension() const = 0;
	virtual std::size_t getElementCount() const = 0;
	virtual const double* getNodeCoordinates(std::size_t) const = 0;
	virtual void getAllElementIDs(const std::string& blockID,std::vector<panzer::LocalOrdinal> & elements) const=0;
	virtual void getAllEdgeSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllEdgeSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllNodeSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllNodeSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllFaceSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllFaceSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllSideSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllSideSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	
	virtual void getMySideSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getMySideSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getMyNodeSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getMyNodeSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getMyEdgeSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getMyEdgeSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getMyFaceSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getMyFaceSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	
	virtual std::vector<panzer::LocalOrdinal> getSideToElementsMap(const std::string&) const=0;
	
	// PhysicsBlock each elements correspond to
	std::vector< Teuchos::RCP<panzer::PhysicsBlock> > elementPhysics;
};

}

#endif
