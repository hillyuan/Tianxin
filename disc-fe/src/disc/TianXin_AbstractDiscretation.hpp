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

#include <cstddef>

namespace TianXin {

struct AbstractDiscretation {
	virtual const double* getNodeCoordinates(std::size_t) const = 0;
	virtual void getAllEdgeSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllEdgeSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllNodeSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllNodeSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllFaceSetIds(const std::string&, std::vector<std::size_t>&) const=0;
	virtual void getAllFaceSetIds(const std::string&, const std::string&, std::vector<std::size_t>&) const=0;
};

}

#endif
